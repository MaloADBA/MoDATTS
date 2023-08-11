from collections import (defaultdict,
                         OrderedDict)
from contextlib import nullcontext
import functools
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.cuda.amp import autocast
import torch.nn.functional as F
from fcn_maker.loss import dice_loss
from .common.network.basic import grad_norm
from .common.losses import (bce,
                            cce,
                            dist_ratio_mse_abs,
                            gan_objective,
                            mae,
                            mse)
from .common.mine import mine


def clear_grad(optimizer):
    # Sets `grad` to None instead of zeroing it.
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad = None


def _reduce(loss):
    # Reduce torch tensors with dim > 1 with a mean on all but the first
    # (batch) dimension. Else, return as is.
    def _mean(x):
        if not isinstance(x, torch.Tensor) or x.dim()<=1:
            return x
        else:
            return x.view(x.size(0), -1).mean(1)
    if not hasattr(loss, '__len__'): loss = [loss]
    if len(loss)==0: return 0
    return sum([_mean(v) for v in loss])


def _cce(p, t):
    # Cross entropy loss that can handle multi-scale classifiers.
    # 
    # If p is a list, process every element (and reduce to batch dim).
    # Each tensor is reduced by `mean` and reduced tensors are averaged
    # together.
    # (For multi-scale classifiers. Each scale is given equal weight.)
    if not isinstance(p, torch.Tensor):
        return sum([_cce(elem, t) for elem in p])/float(len(p))
    # Convert target list to torch tensor (batch_size, 1, 1, 1).
    t = torch.Tensor(t).reshape(-1,1,1).expand(-1,p.size(2),p.size(3)).long()
    if p.is_cuda:
        t = t.to(p.device)
    # Cross-entropy.
    out = F.cross_entropy(p, t)
    # Return if no dimensions beyond batch dim.
    if out.dim()<=1:
        return out
    # Else, return after reducing to batch dim.
    return out.view(out.size(0), -1).mean(1)    # Reduce to batch dim.


def autocast_if_needed():
    # Decorator. If the method's object has a scaler, use the pytorch
    # autocast context; else, run the method without any context.
    def decorator(method):
        @functools.wraps(method)
        def context_wrapper(cls, *args, **kwargs):
            if cls.scaler is not None:
                with torch.cuda.amp.autocast():
                    return method(cls, *args, **kwargs)
            return method(cls, *args, **kwargs)
        return context_wrapper
    return decorator

class segmentation_model(nn.Module):
    """
    Interface wrapper around the `DataParallel` parts of the model.
    """
    def __init__(self, encoder_source, encoder_target, decoder_residual_target, decoder_residual_source,
                 decoder_source, decoder_target,
                 disc_S, disc_T, scaler=None, loss_rec=mae, 
                 loss_seg=None, loss_gan='hinge',
                 num_disc_updates=1, relativistic=False, grad_penalty=None,
                 disc_clip_norm=None,gen_clip_norm=None,  
                 lambda_seg=1, 
                 lambda_mod_cyc=1, lambda_mod_disc=1,
                 debug_ac_gan=False, rng=None):
        super(segmentation_model, self).__init__()
        lambdas = OrderedDict((
            ('lambda_seg',        lambda_seg),
            ('lambda_mod_disc',        lambda_mod_disc),
            ('lambda_mod_cyc',        lambda_mod_cyc)
            ))
        kwargs = OrderedDict((
            ('rng',               rng if rng else np.random.RandomState()),
            ('encoder_source',           encoder_source),
            ('encoder_target',           encoder_target),
            ('decoder_residual_source',    decoder_residual_source),            
            ('decoder_residual_target',  decoder_residual_target),
            ('decoder_source',  decoder_source),
            ('decoder_target',  decoder_target),
            ('scaler',            scaler),
            ('loss_rec',          loss_rec),
            ('loss_seg',          loss_seg if loss_seg else dice_loss()),
            ('loss_gan',          loss_gan),
            ('num_disc_updates',  num_disc_updates),
            ('relativistic',      relativistic),
            ('grad_penalty',      grad_penalty),
            ('gen_clip_norm',     gen_clip_norm),
            ('disc_clip_norm',    disc_clip_norm),
            ('gan_objective',     gan_objective(loss_gan,
                                                relativistic=relativistic,
                                                grad_penalty_real=grad_penalty,
                                                grad_penalty_fake=None,
                                                grad_penalty_mean=0)),
            ('debug_ac_gan',      debug_ac_gan)
            ))
        self.separate_networks = OrderedDict((
            ('disc_S',             disc_S),
            ('disc_T',             disc_T)
            ))
        kwargs.update(lambdas)
        for key, val in kwargs.items():
            setattr(self, key, val)

        # Separate networks not stored directly as attributes.
        # -> Separate parameters, separate optimizers.
        kwargs.update(self.separate_networks)
        
        # Module to compute all network outputs (except discriminator) on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_forward = ['encoder_source', 'encoder_target', 'decoder_residual_source', 
                        'decoder_residual_target',
                        'decoder_source', 'decoder_target','segmenter', 
                        'scaler', 'rng']
        kwargs_forward = dict([(key, val) for key, val in kwargs.items()
                               if key in keys_forward])
        self._forward = _forward(**kwargs_forward, **lambdas)
        if torch.cuda.device_count()>1:
            self._forward = nn.DataParallel(self._forward, output_device=-1)
        
        # Module to compute discriminator losses on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_D = ['gan_objective',
                  'disc_S', 'disc_T', 'scaler', 'debug_ac_gan']
        kwargs_D = dict([(key, val) for key, val in kwargs.items()
                         if key in keys_D])
        self._loss_D = _loss_D(**kwargs_D, **lambdas)
        if torch.cuda.device_count()>1:
            self._loss_D = nn.DataParallel(self._loss_D, output_device=-1)
        
        # Module to compute generator updates on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_G = ['gan_objective',
                  'disc_S', 'disc_T', 'scaler', 'loss_rec', 'debug_ac_gan']
        kwargs_G = dict([(key, val) for key, val in kwargs.items()
                         if key in keys_G])
        self._loss_G = _loss_G(**kwargs_G, **lambdas)
        if torch.cuda.device_count()>1:
            self._loss_G = nn.DataParallel(self._loss_G, output_device=-1)
    
    def _autocast_if_needed(self):
        # If a scaler is passed, use pytorch gradient autocasting. Else,
        # just use a null context that does nothing.
        if self.scaler is not None:
            context = torch.cuda.amp.autocast()
        else:
            context = nullcontext()
        return context
    
    def forward(self, x_SA, x_TA, mask_S=None, optimizer=None, rng=None):
        # Compute gradients and update?
        do_updates_bool = True if optimizer is not None else False
        
        # Apply scaler for gradient backprop if it is passed.
        def backward(loss):
            if self.scaler is not None:
                return self.scaler.scale(loss).backward()
            return loss.backward()
        
        # Apply scaler for optimizer step if it is passed.
        def step(optimizer):
            if self.scaler is not None:
                self.scaler.step(optimizer)
            else:
                optimizer.step()
        
        # Compute all outputs.
        with torch.set_grad_enabled(do_updates_bool):
            with self._autocast_if_needed():
                visible = self._forward(x_SA=x_SA, x_TA=x_TA, rng=rng)
        #####
        # Evaluate discriminator loss and update.
        loss_disc = defaultdict(int)
        loss_D = gradnorm_D = 0
        
        
        if self.lambda_mod_disc:
            for i in range(self.num_disc_updates):
                # Evaluate.
                with torch.set_grad_enabled(do_updates_bool):
                    with self._autocast_if_needed():
                        loss_disc = self._loss_D(
                            x_SA=x_SA, x_TA=x_TA,
                            out_SAT=visible['x_SAT'],
                            out_TAS=visible['x_TAS'])
                        loss_D = _reduce(loss_disc.values())
                # Update discriminator
                disc_S = self.separate_networks['disc_S']
                disc_T = self.separate_networks['disc_T']
                if do_updates_bool:
                    clear_grad(optimizer['D'])
                    with self._autocast_if_needed():
                        _loss = loss_D.mean()
                    backward(_loss)
                    if self.disc_clip_norm:
                        if self.scaler is not None:
                            self.scaler.unscale_(optimizer['D'])
                        nn.utils.clip_grad_norm_(disc_T.parameters(),
                                                 max_norm=self.disc_clip_norm)
                        nn.utils.clip_grad_norm_(disc_S.parameters(),
                                                 max_norm=self.disc_clip_norm)
                    step(optimizer['D'])

        # Evaluate generator losses.
        gradnorm_G = 0
        with torch.set_grad_enabled(do_updates_bool):
            with self._autocast_if_needed():
                losses_G = self._loss_G(x_SA=x_SA, x_TA=x_TA,
                                        x_SAT=visible['x_SAT'],
                                        x_TAS=visible['x_TAS'], 
                                        x_SATS=visible['x_SATS'],
                                        x_TAST=visible['x_TAST'])

        
        # Compute segmentation loss outside of DataParallel modules,
        # avoiding various issues:
        # - scatter of small batch sizes can lead to empty tensors
        # - tracking mask indices is very messy
        # - Dice loss reduced before being returned; then, averaged over GPUs
        mask_S_packed = x_SAM_packed = x_SATM_packed = None
        if mask_S is not None:
            # Prepare a mask Tensor without None entries.
            mask_S_indices = [i for i, m in enumerate(mask_S) if m is not None]
            mask_tumor_indices_S = [i for i in mask_S_indices if np.any(mask_S[i]>0)]
            mask_S_packed = np.array([(mask_S[i]>0)*1 for i in mask_S_indices])
            mask_S_packed = Variable(torch.from_numpy(mask_S_packed))
            if torch.cuda.device_count()==1:
                # `DataParallel` does not respect `output_device` when
                # there is only one GPU. So it returns outputs on GPU rather
                # than CPU, as requested. When this happens, putting mask
                # on GPU allows all values to stay on one device.
                mask_S_packed = mask_S_packed.cuda()

        loss_seg_M = 0.
        loss_seg_trans_M = 0.
                    
        x_SAM_packed = visible['x_SAM'][mask_S_indices]
        
        x_SATM_packed = visible['x_SATM'][mask_S_indices]
        
        loss_seg_trans_M = self.lambda_seg*self.loss_seg(x_SATM_packed[mask_tumor_indices_S], mask_S_packed[mask_tumor_indices_S])
        
        loss_seg_M = self.lambda_seg*self.loss_seg(x_SAM_packed[mask_tumor_indices_S], mask_S_packed[mask_tumor_indices_S])
        
        x_SATM_packed = ((visible['x_SATM'][mask_S_indices] > 0.5)*1)
        x_SAM_packed = ((visible['x_SAM'][mask_S_indices] > 0.5)*1)
        x_TAM_packed = ((visible['x_TAM'] > 0.5)*1)        

        # Include segmentation loss with generator losses and update.
        with self._autocast_if_needed():
            losses_G['l_seg'] = _reduce([loss_seg_M, loss_seg_trans_M])
            losses_G['l_G'] += losses_G['l_seg']
            loss_G = losses_G['l_G']
        if do_updates_bool and isinstance(loss_G, torch.Tensor):
            if 'S' in optimizer:
                clear_grad(optimizer['S'])
            clear_grad(optimizer['G'])
            with self._autocast_if_needed():
                _loss = loss_G.mean()
            backward(_loss)
            if self.scaler is not None:
                self.scaler.unscale_(optimizer['G'])
                if 'S' in optimizer:
                    self.scaler.unscale_(optimizer['S'])
            if self.gen_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.parameters(),
                                         max_norm=self.gen_clip_norm)
            step(optimizer['G'])
            if 'S' in optimizer:
                step(optimizer['S'])
            gradnorm_G = grad_norm(self)
        
        # Unscale norm.
        if self.scaler is not None and do_updates_bool:
            gradnorm_D /= self.scaler.get_scale()
            gradnorm_G /= self.scaler.get_scale()
        
        # Update scaler.
        if self.scaler is not None and do_updates_bool:
            self.scaler.update()
        
        # Compile ouputs.
        outputs = OrderedDict()
        outputs['x_SM'] = mask_S_packed
        outputs.update(visible)
        outputs['x_SATM'] = x_SATM_packed
        outputs['x_SAM'] = x_SAM_packed
        outputs['x_TAM'] = x_TAM_packed
        outputs.update(losses_G)
        outputs['l_DS'] = loss_disc['S']
        outputs['l_DT'] = loss_disc['T']
        
        return outputs


class _forward(nn.Module):
    def __init__(self, encoder_source, encoder_target, decoder_residual_source, 
                 decoder_residual_target,
                 decoder_source, decoder_target,
                 decoder_autoencode=None, scaler=None,
                 lambda_seg=1, lambda_mod_cyc=1, lambda_mod_disc=1,
                 rng=None):
        super(_forward, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.encoder_source            = encoder_source
        self.encoder_target            = encoder_target
        self.decoder_residual_source     = decoder_residual_source       
        self.decoder_residual_target   = decoder_residual_target
        self.decoder_source     = decoder_source 
        self.decoder_target     = decoder_target
        self.decoder_autoencode = decoder_autoencode
        self.scaler             = scaler
        self.lambda_seg         = lambda_seg
        self.lambda_mod_disc    = lambda_mod_disc
        self.lambda_mod_cyc     = lambda_mod_cyc


    
    @autocast_if_needed()
    def forward(self, x_SA, x_TA, rng=None):
         
        # Helper function for summing either two tensors or pairs of tensors
        # across two lists of tensors.
        def add(a, b):
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                return a+b
            else:
                assert not isinstance(a, torch.Tensor)
                assert not isinstance(b, torch.Tensor)
                return [elem_a+elem_b for elem_a, elem_b in zip(a, b)]
        
        # Helper function to split an output into the final image output
        # tensor and a list of intermediate tensors.
        def unpack(x):
            x_list = None
            if not isinstance(x, torch.Tensor):
                x, x_list = x[-1], x[:-1]
            return x, x_list
        
        # SA pathway (sick source)
        s_SA, _, skip_SA = self.encoder_source(x_SA)
        x_SAT = self.decoder_target(s_SA, skip_SA)
        
        s_SAT, _, skip_SAT = self.encoder_target(x_SAT)
        x_SATS = self.decoder_source(s_SAT, skip_SAT)
        
        # TA pathway (sick target)
        s_TA, _, skip_TA = self.encoder_target(x_TA)
        x_TAS = self.decoder_source(s_TA, skip_TA)
        
        s_TAS, _, skip_TAS = self.encoder_source(x_TAS)
        x_TAST = self.decoder_target(s_TAS, skip_TAS)

        # Segment.
        x_SATM = x_SAM = x_TAM = None

        #    # Re-use residual decoder in mode 2.
        x_SATM = self.decoder_residual_target(s_SAT, skip_SAT)
        x_SAM = self.decoder_residual_source(s_SA, skip_SA)
        x_TAM = self.decoder_residual_target(s_TA, skip_TA)

        # Compile outputs and return.
        visible = OrderedDict((
               
            ('x_SA',           x_SA),
            ('x_SAM',        x_SAM),            
            ('x_SAT',          x_SAT),
            ('x_SATM',        x_SATM),
            ('x_SATS',          x_SATS),
            
            ('x_TA',           x_TA),
            ('x_TAM',          x_TAM),
            ('x_TAS',         x_TAS),    
            ('x_TAST',          x_TAST)
            
            ))

        return visible


class _loss_D(nn.Module):
    def __init__(self, gan_objective, disc_T, disc_S, 
                 scaler=None, lambda_seg=1, lambda_mod_cyc=1, lambda_mod_disc=1, debug_ac_gan=False):
        super(_loss_D, self).__init__()
        self._gan               = gan_objective
        self.scaler             = scaler
        self.lambda_seg         = lambda_seg
        self.lambda_mod_disc    = lambda_mod_disc
        self.lambda_mod_cyc     = lambda_mod_cyc
        self.debug_ac_gan       = debug_ac_gan
        self.net = {'disc_T'    : disc_T,
                    'disc_S'    : disc_S}  # Separate params.
    
    @autocast_if_needed()
    def forward(self, x_SA, x_TA,
                out_SAT,
                out_TAS):

        # Detach all tensors; updating discriminator, not generator.
            
        if isinstance(out_SAT, list):
            out_SAT = [x.detach() for x in out_SAT]
        else:
            out_SAT = out_SAT.detach()
            
        if isinstance(out_TAS, list):
            out_TAS = [x.detach() for x in out_TAS]
        else:
            out_TAS = out_TAS.detach()
        
        # Discriminators.
        kwargs_real = None 
        kwargs_fake = None 
        loss_disc = OrderedDict()

        if self.lambda_mod_disc :    
            loss_disc['S'] = self.lambda_mod_disc*self._gan.D(self.net['disc_S'],
                                      fake=out_TAS,
                                      real=x_SA,
                                      kwargs_real=kwargs_real,
                                      kwargs_fake=kwargs_fake,
                                      scaler=self.scaler)

          
            loss_disc['T'] = self.lambda_mod_disc*self._gan.D(self.net['disc_T'],
                                      fake=out_SAT,
                                      real=x_TA,
                                      kwargs_real=kwargs_real,
                                      kwargs_fake=kwargs_fake,
                                      scaler=self.scaler)
            
        return loss_disc


class _loss_G(nn.Module):
    def __init__(self, gan_objective, disc_T, disc_S, scaler=None,
                 loss_rec=mae, lambda_seg=1, 
                 lambda_mod_cyc=1, lambda_mod_disc=1,
                 debug_ac_gan=False):
        super(_loss_G, self).__init__()
        self._gan               = gan_objective
        self.scaler             = scaler
        self.loss_rec           = loss_rec
        self.lambda_seg         = lambda_seg
        self.lambda_mod_disc    = lambda_mod_disc
        self.lambda_mod_cyc     = lambda_mod_cyc
        self.debug_ac_gan       = debug_ac_gan
        self.net = {'disc_T'    : disc_T,
                    'disc_S'    : disc_S}  # Separate params.
    
    @autocast_if_needed()
    def forward(self, x_SA, x_TA,
                x_SAT, x_SATS,
                x_TAS, x_TAST):
                                     

        # Generator loss.
        loss_gen = defaultdict(int)
        kwargs_real = None
        kwargs_fake = None    
                          
        if self.lambda_mod_disc:
            loss_gen['S'] = self.lambda_mod_disc*self._gan.G(self.net['disc_S'],
                          fake=x_TAS,
                          real=x_SA,
                          kwargs_real=kwargs_real,
                          kwargs_fake=kwargs_fake)

            loss_gen['T'] = self.lambda_mod_disc*self._gan.G(self.net['disc_T'],
                          fake=x_SAT,
                          real=x_TA,
                          kwargs_real=kwargs_real,
                          kwargs_fake=kwargs_fake)

        # Reconstruction loss.
        loss_rec = defaultdict(int)
                        
        if self.lambda_mod_cyc:
            loss_rec['x_SATS'] = self.lambda_mod_cyc*self.loss_rec(x_SA, x_SATS)
            loss_rec['x_TAST'] = self.lambda_mod_cyc*self.loss_rec(x_TA, x_TAST)
            
        # All generator losses combined.
        loss_G = ( _reduce(loss_gen.values())
                  +_reduce(loss_rec.values()))
        
        # Compile outputs and return.
        losses = OrderedDict((
            ('l_G',           loss_G),
            ('l_gen_S',      loss_gen['S']),
            ('l_gen_T',      loss_gen['T']),
            ('l_rec_img',         _reduce([loss_rec['x_SATS'], loss_rec['x_TAST']]))
            ))
        return losses

