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
            return x.contiguous().view(x.size(0), -1).mean(1)
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
    def __init__(self, encoder, decoder_common, decoder_residual, segmenter,
                 disc_TA, disc_TB, shape_sample, scaler=None, loss_rec=mae, 
                 loss_seg=None, loss_gan='hinge',
                 num_disc_updates=1, relativistic=False, grad_penalty=None,
                 disc_clip_norm=None,gen_clip_norm=None,  
                 lambda_disc=1, lambda_x_id=1, lambda_z_id=1, lambda_seg=1, 
                 debug_ac_gan=False, rng=None):
        super(segmentation_model, self).__init__()
        lambdas = OrderedDict((
            ('lambda_disc',       lambda_disc),
            ('lambda_x_id',       lambda_x_id),
            ('lambda_z_id',       lambda_z_id),
            ('lambda_seg',        lambda_seg)
            ))
        kwargs = OrderedDict((
            ('rng',               rng if rng else np.random.RandomState()),
            ('encoder',           encoder),
            ('decoder_common',    decoder_common),            
            ('decoder_residual',  decoder_residual),
            ('segmenter',         segmenter),            
            ('shape_sample',      shape_sample),
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
            ('disc_TA',            disc_TA),
            ('disc_TB',            disc_TB)
            ))
        kwargs.update(lambdas)
        for key, val in kwargs.items():
            setattr(self, key, val)

        # Separate networks not stored directly as attributes.
        # -> Separate parameters, separate optimizers.
        kwargs.update(self.separate_networks)
        
        # Module to compute all network outputs (except discriminator) on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_forward = ['encoder', 'decoder_common', 
                        'decoder_residual',
                        'segmenter', 
                        'shape_sample',
                        'scaler', 'rng']
        kwargs_forward = dict([(key, val) for key, val in kwargs.items()
                               if key in keys_forward])
        self._forward = _forward(**kwargs_forward, **lambdas)
        if torch.cuda.device_count()>1:
            self._forward = nn.DataParallel(self._forward, output_device=-1)
        
        # Module to compute discriminator losses on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_D = ['gan_objective', 'disc_TA', 'disc_TB',
                  'scaler', 'debug_ac_gan']
        kwargs_D = dict([(key, val) for key, val in kwargs.items()
                         if key in keys_D])
        self._loss_D = _loss_D(**kwargs_D, **lambdas)
        if torch.cuda.device_count()>1:
            self._loss_D = nn.DataParallel(self._loss_D, output_device=-1)
        
        # Module to compute generator updates on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_G = ['gan_objective', 'disc_TA', 'disc_TB',
                  'scaler', 'loss_rec', 'debug_ac_gan']
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
    
    def forward(self, x_TB, x_TA, x_SAT, mask_S=None, mask_T=None, optimizer=None, rng=None):
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
                visible, hidden = self._forward(x_SAT=x_SAT, x_TA=x_TA, x_TB=x_TB,
                                                               rng=rng)
        #####
        # Evaluate discriminator loss and update.
        loss_disc = defaultdict(int)
        loss_D = gradnorm_D = 0
        if self.lambda_disc :
                
                  
            for i in range(self.num_disc_updates):
                # Evaluate.
                with torch.set_grad_enabled(do_updates_bool):
                    with self._autocast_if_needed():
                        loss_disc = self._loss_D(
                            x_SAT=x_SAT, x_TA=x_TA, x_TB=x_TB, 
                            out_TATB=visible['x_TATB'], out_TBTA=visible['x_TBTA'])
                        loss_D = _reduce(loss_disc.values())
                # Update discriminator
                disc_TA = self.separate_networks['disc_TA']
                disc_TB = self.separate_networks['disc_TB']    
                if do_updates_bool:
                    clear_grad(optimizer['D'])
                    with self._autocast_if_needed():
                        _loss = loss_D.mean()
                    backward(_loss)
                    if self.disc_clip_norm:
                        if self.scaler is not None:
                            self.scaler.unscale_(optimizer['D'])
                        nn.utils.clip_grad_norm_(disc_TA.parameters(),
                                                 max_norm=self.disc_clip_norm)
                        nn.utils.clip_grad_norm_(disc_TB.parameters(),
                                                 max_norm=self.disc_clip_norm)
                    step(optimizer['D'])
                    gradnorm_D = grad_norm(disc_TB)+grad_norm(disc_TA)

        
        # Evaluate generator losses.
        gradnorm_G = 0
        with torch.set_grad_enabled(do_updates_bool):
            with self._autocast_if_needed():
                losses_G = self._loss_G(x_TA=x_TA, x_TB=x_TB, 
                                        x_TATB=visible['x_TATB'], x_TATA=visible['x_TATA'], 
                                        x_TBTB=visible['x_TBTB'], x_TBTA=visible['x_TBTA'], 
                                        
                                        c_TA=hidden['c_TA'], u_TA=hidden['u_TA'], 
                                        c_TATB=hidden['c_TATB'], c_TATA=hidden['c_TATA'], u_TATA=hidden['u_TATA'],
                                        
                                        c_TB=hidden['c_TB'], u_TB=hidden['u_TB'],
                                        u_TB_sampled=hidden['u_TB_sampled'],
                                        c_TBTB=hidden['c_TBTB'], c_TBTA=hidden['c_TBTA'], u_TBTA=hidden['u_TBTA'])

        
        # Compute segmentation loss outside of DataParallel modules,
        # avoiding various issues:
        # - scatter of small batch sizes can lead to empty tensors
        # - tracking mask indices is very messy
        # - Dice loss reduced before being returned; then, averaged over GPUs
        mask_S_packed = mask_T_packed = x_TAM_packed = x_SATM_packed = None
        if mask_S is not None:
            # Prepare a mask Tensor without None entries.
            mask_S_indices = [i for i, m in enumerate(mask_S) if m is not None]
            mask_S_packed = np.array([(mask_S[i]>0)*1 for i in mask_S_indices])
            mask_S_packed = Variable(torch.from_numpy(mask_S_packed))
            if torch.cuda.device_count()==1:
                # `DataParallel` does not respect `output_device` when
                # there is only one GPU. So it returns outputs on GPU rather
                # than CPU, as requested. When this happens, putting mask
                # on GPU allows all values to stay on one device.
                mask_S_packed = mask_S_packed.cuda()
                
        if mask_T is not None:
            # Prepare a mask Tensor without None entries.
            mask_T_indices = [i for i, m in enumerate(mask_T) if m is not None]
            mask_T_packed = np.array([(mask_T[i]>0)*1 for i in mask_T_indices])
            mask_T_packed = Variable(torch.from_numpy(mask_T_packed))
            if torch.cuda.device_count()==1:
                # `DataParallel` does not respect `output_device` when
                # there is only one GPU. So it returns outputs on GPU rather
                # than CPU, as requested. When this happens, putting mask
                # on GPU allows all values to stay on one device.
                mask_T_packed = mask_T_packed.cuda()

        loss_seg_trans_M = 0.
        loss_seg_M = 0.
        
        if self.lambda_seg and mask_T_packed is not None and len(mask_T_packed):
            if self.lambda_seg and mask_S_packed is not None and len(mask_S_packed):
                with self._autocast_if_needed():
                    
                    x_TAM_packed = visible['x_TAM'][mask_T_indices]
                    
                    x_SATM_packed = visible['x_SATM'][mask_S_indices]
                    
                    loss_seg_trans_M = self.lambda_seg*self.loss_seg(x_SATM_packed, mask_S_packed)
                    
                    loss_seg_M = self.lambda_seg*self.loss_seg(x_TAM_packed, mask_T_packed)
                    
                    x_SATM_packed = ((visible['x_SATM'][mask_S_indices] > 0.5)*1)
                    x_TAM_packed = ((visible['x_TAM'][mask_T_indices] > 0.5)*1)
                
            else :
                with self._autocast_if_needed():
                
                    x_TAM_packed = visible['x_TAM'][mask_T_indices]
                           
                    loss_seg_M = self.lambda_seg*self.loss_seg(x_TAM_packed,mask_T_packed)
                    
                    x_TAM_packed = ((visible['x_TAM'][mask_T_indices] > 0.5)*1)   
        
        if not (self.lambda_seg and mask_T_packed is not None and len(mask_T_packed)):
            with self._autocast_if_needed():
                
                    x_SATM_packed = visible['x_SATM'][mask_S_indices]
                           
                    loss_seg_trans_M = self.lambda_seg*self.loss_seg(x_SATM_packed,mask_S_packed)

                    x_SATM_packed = ((visible['x_SATM'][mask_S_indices] > 0.5)*1)  
        
        
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
        outputs['x_TM'] = mask_T_packed
        outputs.update(visible)
        outputs['x_SATM'] = x_SATM_packed
        outputs['x_TAM_prob'] = visible['x_TAM']
        outputs['x_TAM'] = x_TAM_packed
        outputs.update(losses_G)
        outputs['l_D']  = loss_D
        outputs['l_DTA'] = _reduce([loss_disc['TA']])
        outputs['l_DTB'] = _reduce([loss_disc['TB']])
        outputs['l_gradnorm_D'] = gradnorm_D
        outputs['l_gradnorm_G'] = gradnorm_G
        
        return outputs


class _forward(nn.Module):
    def __init__(self, encoder, decoder_common, 
                 decoder_residual, segmenter,
                 shape_sample, decoder_autoencode=None, scaler=None,
                 lambda_disc=1, lambda_x_id=1, lambda_z_id=1, lambda_seg=1, 
                 rng=None):
        super(_forward, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.encoder           = encoder
        self.decoder_common     = decoder_common     
        self.decoder_residual   = decoder_residual
        self.segmenter          = [segmenter]   # Separate params.
        self.shape_sample       = shape_sample
        self.decoder_autoencode = decoder_autoencode
        self.scaler             = scaler
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_seg         = lambda_seg
    
    def _z_sample(self, shape, rng=None):
        if rng is None:
            rng = self.rng
        sample = rng.randn(*tuple(shape)).astype(np.float32)
        ret = Variable(torch.from_numpy(sample))
        ret = ret.to(torch.cuda.current_device())
        return ret

    
    @autocast_if_needed()
    def forward(self, x_SAT, x_TA, x_TB, rng=None):
        
        batch_size = len(x_TA)
        
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
        
        # SAT pathway (transfered sick source)
        s_SAT, skip_SAT, map_SAT = self.encoder(x_SAT) 
        x_SAT_residual = self.decoder_residual(s_SAT, skip_SAT, map_SAT)

        # TA pathway (sick target)
        s_TA, skip_TA, map_TA = self.encoder(x_TA)
        c_TA, u_TA = torch.split(s_TA, [s_TA.size(1)*3//4, s_TA.size(1)*1//4], dim=1)
           
        x_TA_residual = self.decoder_residual(torch.cat([c_TA, u_TA], dim=1), skip_TA, map_TA)
        x_TATB = self.decoder_common(c_TA, skip_TA, map_TA)
        x_TATA = add(x_TATB, x_TA_residual)
        
        s_TATB, _, _ = self.encoder(x_TATB)
        c_TATB, _ = torch.split(s_TATB, [s_TATB.size(1)*3//4, s_TATB.size(1)*1//4], dim=1)
        s_TATA, _, _ = self.encoder(x_TATA)
        c_TATA, u_TATA = torch.split(s_TATA, [s_TATA.size(1)*3//4, s_TATA.size(1)*1//4], dim=1)    

        # TB pathway (healthy target)
        s_TB, skip_TB, map_TB = self.encoder(x_TB)
        
        u_TB_sampled = self._z_sample(u_TA.shape, rng=rng)
        c_TB, u_TB  = torch.split(s_TB, [s_TB.size(1)*3//4, s_TB.size(1)*1//4], dim=1)
        
        x_TB_residual = self.decoder_residual(torch.cat([c_TB, u_TB_sampled], dim=1), skip_TB, map_TB)
        x_TBTB = self.decoder_common(c_TB, skip_TB, map_TB)
        x_TBTA = add(x_TBTB, x_TB_residual)
        
        s_TBTB, _, _ = self.encoder(x_TBTB)
        c_TBTB, _ = torch.split(s_TBTB, [s_TBTB.size(1)*3//4, s_TBTB.size(1)*1//4], dim=1)
        s_TBTA, _, _ = self.encoder(x_TBTA)
        c_TBTA, u_TBTA = torch.split(s_TBTA, [s_TBTA.size(1)*3//4, s_TBTA.size(1)*1//4], dim=1)

        # Segment.
        x_SATM = x_TAM = None
        if self.lambda_seg:
            if self.segmenter[0] is not None:
                # Use separate segmentation decoder in mode 1.
                x_SATM = self.segmenter[0](s_SAT, skip_SAT, map_SAT, mode=1)
                x_TAM = self.segmenter[0](s_TA, skip_TA, map_TA, mode=1)           
            else:
                # Re-use residual decoder in mode 1.
                x_SATM = self.decoder_residual(s_SAT, skip_SAT, map_SAT, mode=1)
                x_TAM = self.decoder_residual(s_TA, skip_TA, map_TA, mode=1)
        
        # Compile outputs and return.
        visible = OrderedDict((
               
            ('x_SAT',           x_SAT),
            ('x_SAT_residual', x_SAT_residual),
            ('x_SATM',        x_SATM),
            
            ('x_TA',           x_TA),
            ('x_TATB',          x_TATB),
            ('x_TA_residual', x_TA_residual),
            ('x_TAM',          x_TAM),
            ('x_TATA',          x_TATA),
            
            ('x_TB',           x_TB),
            ('x_TBTB',          x_TBTB),
            ('x_TB_residual', x_TB_residual),
            ('x_TBTA',          x_TBTA)
            
            ))
        hidden = OrderedDict((
            ('c_TA',          c_TA),
            ('u_TA',          u_TA),
            ('c_TATB',          c_TATB),
            ('c_TATA',          c_TATA),
            ('u_TATA',          u_TATA),
                  
            ('c_TB',          c_TB),
            ('u_TB',          u_TB),
            ('u_TB_sampled',          u_TB_sampled),
            ('c_TBTB',          c_TBTB),
            ('c_TBTA',          c_TBTA),
            ('u_TBTA',          u_TBTA)
            ))
        
        return visible, hidden


class _loss_D(nn.Module):
    def __init__(self, gan_objective, disc_TA, disc_TB, 
                 scaler=None, lambda_disc=1, lambda_x_id=1, lambda_z_id=1, lambda_seg=1, 
                 debug_ac_gan=False):
        super(_loss_D, self).__init__()
        self._gan               = gan_objective
        self.scaler             = scaler
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_seg         = lambda_seg
        self.debug_ac_gan       = debug_ac_gan
        self.net = {'disc_TA'    : disc_TA,
                    'disc_TB'    : disc_TB}  # Separate params.
    
    @autocast_if_needed()
    def forward(self, x_SAT, x_TA, x_TB, 
                out_TATB, out_TBTA):

        # Detach all tensors; updating discriminator, not generator.
            
        if isinstance(out_TATB, list):
            out_TATB = [x.detach() for x in out_TATB]
        else:
            out_TATB = out_TATB.detach()

        if isinstance(out_TBTA, list):
            out_TBTA = [x.detach() for x in out_TBTA]
        else:
            out_TBTA = out_TBTA.detach()
        
        # Discriminators.
        kwargs_real = None 
        kwargs_fake = None 
        loss_disc = OrderedDict()
        if self.lambda_disc or self.lambda_mod_disc:
            
            loss_disc['TA'] = self.lambda_disc*self._gan.D(self.net['disc_TA'],
                                     fake=out_TBTA,
                                     real=x_TA,
                                     kwargs_real=kwargs_real,
                                     kwargs_fake=kwargs_fake,
                                     scaler=self.scaler)
            
            loss_disc['TB'] = self.lambda_disc*self._gan.D(self.net['disc_TB'],
                                     fake=out_TATB,
                                     real=x_TB,
                                     kwargs_real=kwargs_real,
                                     kwargs_fake=kwargs_fake,
                                     scaler=self.scaler)

        return loss_disc


class _loss_G(nn.Module):
    def __init__(self, gan_objective, disc_TA, disc_TB, scaler=None,
                 loss_rec=mae, lambda_disc=1, lambda_x_id=1, lambda_z_id=1, lambda_seg=1, 
                 debug_ac_gan=False):
        super(_loss_G, self).__init__()
        self._gan               = gan_objective
        self.scaler             = scaler
        self.loss_rec           = loss_rec
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_seg         = lambda_seg
        self.debug_ac_gan       = debug_ac_gan
        self.net = {'disc_TA'    : disc_TA,
                    'disc_TB'    : disc_TB}  # Separate params.
    
    @autocast_if_needed()
    def forward(self, x_TA, x_TB, 
                x_TATB, x_TATA,
                x_TBTB, x_TBTA,
                c_TA, u_TA, c_TATB, c_TATA, u_TATA,
                c_TB, u_TB, u_TB_sampled, c_TBTB, c_TBTA, u_TBTA):
                                     

        # Generator loss.
        loss_gen = defaultdict(int)
        kwargs_real = None
        kwargs_fake = None  
              
        if self.lambda_disc:
            loss_gen['TA'] = self.lambda_disc*self._gan.G(self.net['disc_TA'],
                         fake=x_TBTA,
                         real=x_TA,
                         kwargs_real=kwargs_real,
                         kwargs_fake=kwargs_fake)
            loss_gen['TB'] = self.lambda_disc*self._gan.G(self.net['disc_TB'],
                         fake=x_TATB,
                         real=x_TB,
                         kwargs_real=kwargs_real,
                         kwargs_fake=kwargs_fake)
                          
        
        # Reconstruction loss.
        loss_rec = defaultdict(int)
        if self.lambda_x_id:
            loss_rec['x_TATA'] = self.lambda_x_id*self.loss_rec(x_TA, x_TATA)
            loss_rec['x_TBTB'] = self.lambda_x_id*self.loss_rec(x_TB, x_TBTB)
     
        if self.lambda_z_id:
            loss_rec['TATB'] = self.lambda_z_id*self.loss_rec(c_TA, c_TATB)
            loss_rec['TATA'] = self.lambda_z_id*self.loss_rec(torch.cat([c_TA, u_TA], dim=1), torch.cat([c_TATA, u_TATA], dim=1))
            loss_rec['TBTB'] = self.lambda_z_id*self.loss_rec(c_TB, c_TBTB)
            loss_rec['TBTA'] = self.lambda_z_id*self.loss_rec(torch.cat([c_TB, u_TB_sampled], dim=1), torch.cat([c_TBTA, u_TBTA], dim=1))
            
        # All generator losses combined.
        loss_G = ( _reduce(loss_gen.values())
                  +_reduce(loss_rec.values()))
        
        # Compile outputs and return.
        losses = OrderedDict((
            ('l_G',           loss_G),
            ('l_gen_TA',      _reduce([loss_gen['TA']])),
            ('l_gen_TB',      _reduce([loss_gen['TB']])),
            ('l_rec_img',         _reduce([loss_rec['x_TATA'], loss_rec['x_TBTB']])),
            ('l_rec_features',       _reduce([loss_rec['TATB'], loss_rec['TATA'],
                                              loss_rec['TBTB'], loss_rec['TBTA']]))
            ))
        return losses

