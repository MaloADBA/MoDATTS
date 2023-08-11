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

        kwargs.update(lambdas)
        for key, val in kwargs.items():
            setattr(self, key, val)

        
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
                visible = self._forward(x_SAT=x_SAT, x_TA=x_TA, x_TB=x_TB,
                                                               rng=rng)
        
        
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
        
        losses_G = {}
        # Include segmentation loss with generator losses and update.
        with self._autocast_if_needed():
            losses_G['l_seg'] = _reduce([loss_seg_M, loss_seg_trans_M])
            loss_G = losses_G['l_seg']
        if do_updates_bool and isinstance(loss_G, torch.Tensor):
            clear_grad(optimizer['G'])
            with self._autocast_if_needed():
                _loss = loss_G.mean()
            backward(_loss)
            step(optimizer['G'])
        
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
 
        x_SATM = torch.sigmoid(self.decoder_residual(s_SAT, skip_SAT, map_SAT))
        
        # TA pathway (sick target)
        s_TA, skip_TA, map_TA = self.encoder(x_TA)

        x_TAM = torch.sigmoid(self.decoder_residual(s_TA, skip_TA, map_TA))

        
        # Compile outputs and return.
        visible = OrderedDict((
               
            ('x_SAT',           x_SAT),
            ('x_SATM',        x_SATM),
            
            ('x_TA',           x_TA),
            ('x_TAM',          x_TAM)
            
            ))
        return visible
