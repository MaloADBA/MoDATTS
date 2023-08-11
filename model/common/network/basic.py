import torch
from torch import nn
from torch.nn.utils import (spectral_norm,
                            remove_spectral_norm)
from torch.nn import functional as F
import numpy as np
from fcn_maker.blocks import (adjust_to_size,
                              convolution,
                              do_upsample,
                              get_nonlinearity,
                              get_initializer,
                              get_dropout,
                              block_abstract,
                              basic_block,
                              dense_block,
                              identity_block,
                              max_pooling,
                              norm_nlin_conv,
                              repeat_block,
                              shortcut,
                              tiny_block)

def get_output_shape(layer, input_shape):
    """
    Works for `convolution`, `nn.Linear`, `identity_block`, `basic_block`,
    `bottleneck_block`, `tiny_block`, `dense_block`, `pool_block`,
    `repeat_block`.
    
    `input_shape` is without batch dimension.
    """
    def _padding_array(padding):
        if not hasattr(padding, '__len__'):
            return np.array(2*padding)
        arr = [padding[i]+padding[i+1] for i in range(0, len(padding), 2)]
        return np.array(arr)
        
    def compute_conv_out_shape(input_shape, out_channels, padding,
                               kernel_size, stride=1):
        out_shape = 1 + ( np.array(input_shape)[1:]
                         +_padding_array(padding)
                         -np.array(kernel_size)) // np.array(stride)
        out_shape = (out_channels,)+tuple(out_shape)
        return out_shape
    
    def compute_tconv_out_shape(input_shape, out_channels, padding,
                                kernel_size, stride=1):
        out_shape = ( (np.array(input_shape[1:])-1)*np.array(stride)
                     -_padding_array(padding)
                     +np.array(kernel_size))
        out_shape = (out_channels,)+tuple(out_shape)
        return out_shape
    
    def compute_pool_out_shape(input_shape, padding,
                               stride=2, ceil_mode=False):
        input_spatial_shape_padded = ( np.array(input_shape)[1:]
                                      +_padding_array(padding))
        out_shape = input_spatial_shape_padded//stride
        if ceil_mode:
            out_shape += input_spatial_shape_padded%stride
        out_shape = (input_shape[0],)+tuple(out_shape)
        return out_shape
    
    def compute_block_upsample(layer, input_shape, kernel_size=3, stride=1):
        if layer.upsample_mode=='conv':
            out_shape = compute_tconv_out_shape(
                                           input_shape=input_shape,
                                           out_channels=layer.out_channels,
                                           padding=0,
                                           kernel_size=kernel_size,
                                           stride=stride)
        elif layer.upsample_mode=='repeat':
            out_shape = (input_shape[0],)+tuple(np.array(input_shape[1:])*2)
        else:
            raise AssertionError("Invalid `upsample_mode`: {}"
                                 "".format(layer.upsample_mode))
        return out_shape
    
    if isinstance(layer, convolution):
        out_shape = compute_conv_out_shape(input_shape=input_shape,
                                           out_channels=layer.out_channels,
                                           padding=layer.padding,
                                           kernel_size=layer.op.kernel_size,
                                           stride=layer.op.stride)
        return out_shape
    elif isinstance(layer, nn.Linear):
        return (layer.out_features,)
    elif isinstance(layer, identity_block):
        return input_shape
    elif isinstance(layer, basic_block):
        padding = 0
        if layer.conv_padding:
            padding = [(layer.kernel_size-1)//2,
                       (layer.kernel_size-int(layer.subsample))//2]*layer.ndim
        out_shape = compute_conv_out_shape(input_shape=input_shape,
                                           out_channels=layer.out_channels,
                                           padding=padding,
                                           kernel_size=layer.kernel_size,
                                           stride=2 if layer.subsample else 1)
        if layer.upsample:
            out_shape = compute_block_upsample(layer, out_shape)
        out_shape = compute_conv_out_shape(input_shape=out_shape,
                                           out_channels=layer.out_channels,
                                           padding=padding,
                                           kernel_size=layer.kernel_size,
                                           stride=1)
        return out_shape
    elif isinstance(layer, bottleneck_block):
        padding = 0
        if layer.conv_padding:
            padding = [(layer.kernel_size-1)//2,
                       (layer.kernel_size-int(layer.subsample))//2]*layer.ndim
        out_shape = (layer.out_channels//4,)+input_shape[1:]
        out_shape = compute_conv_out_shape(input_shape=out_shape,
                                           out_channels=layer.out_channels//4,
                                           padding=padding,
                                           kernel_size=layer.kernel_size,
                                           stride=2 if layer.subsample else 1)
        out_shape = (layer.out_channels,)+out_shape[1:]
        if layer.upsample:
            out_shape = compute_block_upsample(layer, out_shape)
        out_shape = compute_conv_out_shape(input_shape=out_shape,
                                           out_channels=layer.out_channels,
                                           padding=padding,
                                           kernel_size=layer.kernel_size,
                                           stride=1)
        return out_shape
    elif isinstance(layer, (tiny_block, pool_block)):
        out_shape = input_shape
        if layer.subsample:
            out_shape = compute_pool_out_shape(input_shape=input_shape,
                                               padding=0,
                                               stride=2)
        padding = 0
        if layer.conv_padding:
            padding = [(layer.kernel_size-1)//2,
                       (layer.kernel_size-int(layer.subsample))//2]*layer.ndim
        out_shape = compute_conv_out_shape(input_shape=out_shape,
                                           out_channels=layer.out_channels,
                                           padding=padding,
                                           kernel_size=layer.kernel_size,
                                           stride=1)
        if layer.upsample:
            out_shape = compute_block_upsample(layer, out_shape)
        return out_shape
    elif isinstance(layer, conv_block):
        padding = 0
        if layer.conv_padding:
            padding = [(layer.kernel_size-1)//2,
                       (layer.kernel_size-int(layer.subsample))//2]*layer.ndim
        out_shape = compute_conv_out_shape(input_shape=input_shape,
                                           out_channels=layer.out_channels,
                                           padding=padding,
                                           kernel_size=layer.kernel_size,
                                           stride=2 if layer.subsample else 1)
        if layer.upsample:
            out_shape = compute_block_upsample(layer, out_shape)
        return out_shape
    elif isinstance(layer, dense_block):
        # Setting `conv_padding` to False doesn't make sense.
        out_shape = input_shape
        if layer.subsample:
            out_shape = compute_pool_out_shape(input_shape=input_shape,
                                               padding=0,
                                               stride=2)
        if layer.upsample:
            out_shape = compute_block_upsample(layer, out_shape)
        out_shape = (layer.out_channels,)+out_shape[1:]
        return out_shape
    elif isinstance(layer, repeat_block):
        out_shape = input_shape
        for block in layer.blocks:
            out_shape = get_output_shape(block, out_shape)
        return out_shape
    else:
        raise NotImplementedError("Shape inference not implemented for "
                                  "layer type {}.".format(type(layer)))
    

def batch_normalization(ndim=2, *args, **kwargs):
    if ndim==1:
        return torch.nn.BatchNorm1d(*args, **kwargs)
    if ndim==2:
        return torch.nn.BatchNorm2d(*args, **kwargs)
    elif ndim==3:
        return torch.nn.BatchNorm3d(*args, **kwargs)
    else:
        raise ValueError("ndim must be 1, 2, or 3")
    

def instance_normalization(ndim=2, *args, **kwargs):
    if ndim==1:
        return torch.nn.InstanceNorm1d(*args, **kwargs)
    if ndim==2:
        return torch.nn.InstanceNorm2d(*args, **kwargs)
    elif ndim==3:
        return torch.nn.InstanceNorm3d(*args, **kwargs)
    else:
        raise ValueError("ndim must be 1, 2, or 3")


def group_normalization(num_features, ndim=None, *args, **kwargs):
    return torch.nn.GroupNorm(*args, num_channels=num_features, **kwargs)


class layer_normalization(nn.Module):
    """
    Ming-Yu's layer normalization implementation, to avoid the ridiculous need
    to specify output shape with pytorch's default LayerNorm implementation.
    """
    def __init__(self, num_features, eps=1e-5, affine=True, ndim=None):
        super(layer_normalization, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two 
            # lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class norm_nlin_fc(torch.nn.Module):
    """
    Helper to build a norm -> ReLU -> fully-connected.
    """
    def __init__(self, in_features, out_features, nonlinearity='ReLU',
                 normalization=instance_normalization, norm_kwargs=None,
                 init='kaiming_normal_'):
        super(norm_nlin_fc, self).__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.in_features = in_features
        self.out_features = out_features
        self.nonlinearity = nonlinearity
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.init = init
        if normalization is not None:
            self._modules['norm'] = normalization(
                                                ndim=1,
                                                num_features=self.in_features,
                                                **norm_kwargs)
        self._modules['nlin'] = get_nonlinearity(nonlinearity)
        self._modules['fc'] = nn.Linear(in_features=self.in_features,
                                        out_features=self.out_features)
        if init is not None:
            self._modules['fc'].weight.data = \
                get_initializer(init)(self._modules['fc'].weight.data)

    def forward(self, input):
        out = input
        for op in self._modules.values():
            out = op(out)
        return out


class encoder(nn.Module):
    def __init__(self, input_shape, num_conv_blocks, block_type,
                 num_channels_list, skip=True, dropout=0.,
                 normalization=instance_normalization, norm_kwargs=None,
                 conv_padding=True, padding_mode='constant', kernel_size=3,
                 init='kaiming_normal_', nonlinearity='ReLU', ndim=2):
        super(encoder, self).__init__()
        
        # ndim must be only 2 or 3.
        if ndim not in [2, 3]:
            raise ValueError("`ndim` must be either 2 or 3")
        
        # num_channels should be specified once for every block.
        if len(num_channels_list)!=num_conv_blocks:
            raise ValueError("`num_channels_list` must have the same number "
                             "of entries as there are blocks.")
        
        self.input_shape = input_shape
        self.num_conv_blocks = num_conv_blocks
        self.block_type = block_type
        self.num_channels_list = num_channels_list
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        
        self.in_channels = input_shape[0]
        self.out_channels = self.num_channels_list[-1]
        
        '''
        Set up blocks.
        '''
        self.blocks = nn.ModuleList()
        shape = self.input_shape
        last_channels = self.in_channels
        conv = convolution(in_channels=last_channels,
                           out_channels=self.num_channels_list[0],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           padding_mode=self.padding_mode,
                           init=self.init)
        self.blocks.append(conv)
        shape = get_output_shape(conv, shape)
        last_channels = self.num_channels_list[0]
        for i in range(1, self.num_conv_blocks):
            block = self.block_type(in_channels=last_channels,
                                    num_filters=self.num_channels_list[i],
                                    subsample=True,
                                    skip=skip,
                                    dropout=self.dropout,
                                    normalization=self.normalization,
                                    norm_kwargs=self.norm_kwargs,
                                    conv_padding=self.conv_padding,
                                    padding_mode=self.padding_mode,
                                    kernel_size=self.kernel_size,
                                    init=self.init,
                                    nonlinearity=self.nonlinearity,
                                    ndim=self.ndim)
            self.blocks.append(block)
            shape = get_output_shape(block, shape)
            last_channels = self.num_channels_list[i]
        if normalization is not None:
            block = normalization(ndim=self.ndim,
                                  num_features=last_channels,
                                  **self.norm_kwargs)
            self.blocks.append(block)
        self.blocks.append(get_nonlinearity(self.nonlinearity))
        self.output_shape = shape
            
    def forward(self, input):
        skips = []
        size = input.size()
        out = input
        for m in self.blocks:
            out_prev = out
            out = m(out_prev)
            out_size = out.size()
            if out_size[-1]!=size[-1] or out_size[-2]!=size[-2]:
                # Skip forward feature stacks prior to resolution change.
                size = out_size
                skips.append(out_prev)
        return out, skips
    

class decoder(nn.Module):
    def __init__(self, input_shape, output_shape, num_conv_blocks, block_type,
                 num_channels_list, num_classes=None, skip=True, dropout=0.,
                 normalization=layer_normalization, norm_kwargs=None,
                 conv_padding=True, padding_mode='constant', kernel_size=3,
                 upsample_mode='conv', init='kaiming_normal_',
                 nonlinearity='ReLU', long_skip_merge_mode=None, ndim=2):
        super(decoder, self).__init__()
        
        # ndim must be only 2 or 3.
        if ndim not in [2, 3]:
            raise ValueError("`ndim` must be either 2 or 3")
        
        # num_channels should be specified once for every block.
        if len(num_channels_list)!=num_conv_blocks:
            raise ValueError("`num_channels_list` must have the same number "
                             "of entries as there are blocks.")
        
        # long_skip_merge_mode settings.
        valid_modes = [None, 'skinny_cat', 'cat', 'pool']
        if long_skip_merge_mode not in valid_modes:
            raise ValueError("`long_skip_merge_mode` must be one of {}."
                             "".format(", ".join(["\'{}\'".format(mode)
                                                  for mode in valid_modes])))
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_conv_blocks = num_conv_blocks
        self.block_type = block_type
        self.num_channels_list = num_channels_list
        self.num_classes = num_classes
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.upsample_mode = upsample_mode
        self.init = init
        self.nonlinearity = nonlinearity
        self.long_skip_merge_mode = long_skip_merge_mode
        self.ndim = ndim
        
        self.in_channels  = self.input_shape[0]
        self.out_channels = self.output_shape[0]
        
        # Compute all intermediate conv shapes by working backward from the 
        # output shape.
        self._shapes = [self.output_shape,
                        (self.num_channels_list[-1],)+self.output_shape[1:]]
        for i in range(2, self.num_conv_blocks):
            shape_spatial = np.array(self._shapes[-1][1:])//2
            shape = (self.num_channels_list[-i],)+tuple(shape_spatial)
            self._shapes.append(shape)
        self._shapes.append(self.input_shape)
        self._shapes = self._shapes[::-1]
        
        '''
        Set up blocks.
        '''
        self.cats   = nn.ModuleList()
        self.blocks = nn.ModuleList()
        shape = self.input_shape
        last_channels = shape[0]
        for n in range(self.num_conv_blocks):
            upsample = bool(n<self.num_conv_blocks-1)    # Not on last layer.
            def _select(a, b=None):
                return a if n>0 else b
            block = self.block_type(in_channels=last_channels,
                                    num_filters=self.num_channels_list[n],
                                    upsample=upsample,
                                    upsample_mode=self.upsample_mode,
                                    skip=_select(self.skip, False),
                                    dropout=self.dropout,
                                    normalization=_select(self.normalization),
                                    conv_padding=self.conv_padding,
                                    padding_mode=self.padding_mode,
                                    kernel_size=self.kernel_size,
                                    init=self.init,
                                    nonlinearity=_select(self.nonlinearity),
                                    ndim=self.ndim)
            self.blocks.append(block)
            shape = get_output_shape(block, shape)
            last_channels = self.num_channels_list[n]
            if upsample:
                if   self.long_skip_merge_mode=='skinny_cat':
                    cat = conv_block(in_channels=self.num_channels_list[n+1],
                                     num_filters=1,
                                     skip=False,
                                     normalization=instance_normalization,
                                     nonlinearity=None,
                                     kernel_size=1,
                                     init=self.init,
                                     ndim=self.ndim)
                    self.cats.append(cat)
                    last_channels += 1
                elif self.long_skip_merge_mode=='cat':
                    last_channels *= 2
                else:
                    pass
            
        '''
        Final output - change number of channels.
        '''
        out_kwargs = {'normalization': self.normalization,
                      'norm_kwargs': self.norm_kwargs,
                      'nonlinearity': self.nonlinearity,
                      'padding_mode': self.padding_mode}
        self.out_conv = norm_nlin_conv(in_channels=last_channels,
                                       out_channels=self.out_channels,
                                       kernel_size=self.kernel_size,
                                       init=self.init,
                                       **out_kwargs)
        
    def forward(self, z, skip_info=None):
        out = z
        out_skips = []
        if skip_info is not None:
            skip_info = skip_info[::-1]
        for n, block in enumerate(self.blocks):
            shape_out = self._shapes[n+1]
            out_skips.append(out)
            skip = skip_info[n]
            if (self.long_skip_merge_mode=='pool' and skip_info is not None
                                                  and n<len(skip_info)):
                skip = skip_info[n]
                out = block(out, unpool_indices=skip)
            else:
                out = block(out)
            out = adjust_to_size(out, shape_out[1:])
            if not out.is_contiguous():
                out = out.contiguous()
            if (self.long_skip_merge_mode is not None and skip_info is not None
                                                      and n<len(skip_info)):
                skip = skip_info[n]
                if   self.long_skip_merge_mode=='skinny_cat':
                    cat = self.cats[n]
                    out = torch.cat([out, cat(skip)], dim=1)
                elif self.long_skip_merge_mode=='cat':
                    out = torch.cat([out, skip], dim=1)
                elif self.long_skip_merge_mode=='sum':
                    out = out+skip
                else:
                    raise ValueError("Skip merge mode unrecognized \'{}\'."
                                     "".format(self.long_skip_merge_mode))
        out = self.out_conv(out)
        return out, out_skips

    
class mlp(nn.Module):
    def __init__(self, n_layers, n_input, n_output, n_hidden=None,
                 init='kaiming_normal_'):
        super(mlp, self).__init__()
        assert(n_layers > 0)
        self.n_layers = n_layers
        self.n_input  = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.init = init
        if n_hidden is None:
            self.n_hidden = n_output
        layers = []
        if n_layers > 1:
            layers =  [nn.Linear(in_features=n_input, out_features=n_hidden)]
            layers += [nn.Linear(in_features=n_hidden,
                                 out_features=n_hidden)]*max(0, n_layers-2)
            layers += [nn.Linear(in_features=n_hidden, out_features=n_output)]
        else:
            layers = [nn.Linear(in_features=n_input, out_features=n_output)]
        self.model = nn.Sequential(*tuple(layers))
        for layer in self.model:
            if isinstance(layer, nn.Linear) and init is not None:
                layer.weight.data = get_initializer(init)(layer.weight.data)
    def forward(self, x):
        return self.model(x)


class munit_discriminator(nn.Module):
    def __init__(self, input_dim, num_channels_list, num_scales=3,
                 normalization=None, norm_kwargs=None, kernel_size=5,
                 nonlinearity=lambda:nn.LeakyReLU(0.2, inplace=True),
                 padding_mode='reflect', init='kaiming_normal_', ndim=2):
        super(munit_discriminator, self).__init__()
        self.input_dim = input_dim
        self.num_channels_list = num_channels_list
        self.num_scales = num_scales
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.kernel_size = kernel_size
        self.nonlinearity = nonlinearity
        self.padding_mode = padding_mode
        self.init = init
        self.ndim = ndim
        self.downsample = avg_pooling(3,
                                      stride=2,
                                      padding=1,
                                      count_include_pad=False,
                                      ndim=ndim)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        cnn = []
        layer = convolution(in_channels=self.input_dim,
                            out_channels=self.num_channels_list[0],
                            kernel_size=self.kernel_size,
                            stride=2,
                            padding=(self.kernel_size-1)//2,
                            padding_mode=self.padding_mode,
                            init=self.init,
                            ndim=self.ndim)
        cnn.append(layer)
        for i, (ch0, ch1) in enumerate(zip(self.num_channels_list[:-1],
                                           self.num_channels_list[1:])):
            normalization = self.normalization if i>0 else None
            layer = munit_norm_nlin_conv(
                in_channels=ch0,
                out_channels=ch1,
                kernel_size=self.kernel_size,
                subsample=True,
                conv_padding=True,
                padding_mode=self.padding_mode,
                init=self.init,
                nonlinearity=self.nonlinearity,
                normalization=normalization,
                norm_kwargs=self.norm_kwargs,
                ndim=self.ndim
            )
            cnn.append(layer)
        layer = munit_norm_nlin_conv(
            in_channels=self.num_channels_list[-1],
            out_channels=1,
            kernel_size=1,
            nonlinearity=self.nonlinearity,
            normalization=self.normalization,
            norm_kwargs=self.norm_kwargs,
            ndim=self.ndim
        )
        cnn.append(layer)
        cnn = nn.Sequential(*cnn)
        return cnn

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs


"""
Helper to build a norm -> ReLU -> conv block
This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
"""
class munit_norm_nlin_conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 subsample=False, upsample=False, upsample_mode='repeat',
                 nonlinearity='ReLU', normalization=batch_normalization,
                 norm_kwargs=None, conv_padding=True, padding_mode='constant',
                 init='kaiming_normal_', ndim=2):
        super().__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subsample = subsample
        self.upsample = upsample
        self.upsample_mode = upsample_mode
        self.nonlinearity = nonlinearity
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.padding_mode = padding_mode
        self.init = init
        self.ndim = ndim
        if normalization is not None:
            self._modules['norm'] = normalization(ndim=ndim,
                                                  num_features=in_channels,
                                                  **norm_kwargs)
        self._modules['nlin'] = get_nonlinearity(nonlinearity)
        if upsample:
            self._modules['upsample'] = do_upsample(mode=upsample_mode,
                                                    ndim=ndim,
                                                    in_channels=in_channels,
                                                    out_channels=in_channels,
                                                    kernel_size=2)
        stride = 1
        if subsample:
            stride = 2
        if conv_padding:
            # For odd kernel sizes, equivalent to kernel_size//2.
            # For even kernel sizes, two cases:
            #    (1) [kernel_size//2-1, kernel_size//2] @ stride 1
            #    (2) kernel_size//2 @ stride 2
            # This way, even kernel sizes yield the same output size as
            # odd kernel sizes. When subsampling, even kernel sizes allow
            # possible downscaling without aliasing.
            padding = [(kernel_size-1)//2,
                       (kernel_size-int(subsample))//2]*ndim
        else:
            padding = 0
        self._modules['conv'] = convolution(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            ndim=ndim,
                                            stride=stride,
                                            init=init,
                                            padding=padding,
                                            padding_mode=padding_mode,
                                            autopad=True)
    def forward(self, input):
        out = input
        for op in self._modules.values():
            out = op(out)
        return out


class convolution(torch.nn.Module):
    """
    Select 2D or 3D as argument (ndim) and initialize weights on creation.
    """
    def __init__(self, ndim=2, init=None, padding=None,
                 padding_mode='constant', autopad=False, *args, **kwargs):
        super(convolution, self).__init__()
        if ndim==2:
            conv = torch.nn.Conv2d
        elif ndim==3:
            conv = torch.nn.Conv3d
        else:
            ValueError("ndim must be 2 or 3")
        self.ndim = ndim
        self.init = init
        self.padding = padding
        self.padding_mode = padding_mode
        self.autopad = autopad
        self.op = conv(*args, **kwargs)
        self.in_channels = self.op.in_channels
        self.out_channels = self.op.out_channels
        if init is not None:
            self.op.weight.data = get_initializer(init)(self.op.weight.data)

    def forward(self, input):
        out = input
        if self.padding is not None:
            padding = self.padding
            if not hasattr(padding, '__len__'):
                padding = [self.padding]*self.ndim*2
            padding_mode = self.padding_mode
            size = out.size()[2:]
            if np.any( np.greater_equal(padding[ ::2], size)
                      +np.greater_equal(padding[1::2], size)):
                # Padding size should be less than the corresponding input
                # dimension. Else, use constant.
                padding_mode = 'constant'
            out = F.pad(out, pad=padding, mode=padding_mode, value=0)
        if self.autopad:
            # If the input is smaller than the kernel size, pad it.
            padding = []
            for i in range(self.ndim):
                diff = max(0, self.op.kernel_size[i] - input.size(2+i))
                padding.extend([diff//2, diff//2 + diff%2])
            out = F.pad(out, pad=padding, mode=self.padding_mode, value=0)
        out = self.op(out)
        return out


def max_unpooling(*args, ndim=2, **kwargs):
    if ndim==2:
        return torch.nn.MaxUnpool2d(*args, **kwargs)
    elif ndim==3:
        return torch.nn.MaxUnpool3d(*args, **kwargs)
    else:
        raise ValueError("ndim must be 2 or 3")


def avg_pooling(*args, ndim=2, **kwargs):
    if ndim==2:
        return torch.nn.AvgPool2d(*args, **kwargs)
    elif ndim==3:
        return torch.nn.AvgPool3d(*args, **kwargs)
    else:
        raise ValueError("ndim must be 2 or 3")


"""
Bottleneck architecture for > 34 layer resnet.
Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
"""
class bottleneck_block(block_abstract):
    def __init__(self, in_channels, num_filters, subsample=False,
                 upsample=False, upsample_mode='repeat', skip=True,
                 dropout=0., normalization=batch_normalization,
                 norm_kwargs=None, conv_padding=True, padding_mode='constant',
                 kernel_size=3, init='kaiming_normal_', nonlinearity='ReLU',
                 ndim=2):
        super(bottleneck_block, self).__init__(in_channels, num_filters,
                                               subsample, upsample)
        if norm_kwargs is None:
            norm_kwargs = {}
        self.out_channels = num_filters
        self.upsample_mode = upsample_mode
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        self.op = []
        self.op += [norm_nlin_conv(in_channels=in_channels,
                                   out_channels=num_filters//4,
                                   kernel_size=1,
                                   subsample=subsample,
                                   normalization=normalization,
                                   norm_kwargs=norm_kwargs,
                                   conv_padding=conv_padding,
                                   padding_mode=padding_mode,
                                   init=init,
                                   nonlinearity=nonlinearity,
                                   ndim=ndim)]
        self.op += [norm_nlin_conv(in_channels=num_filters//4,
                                   out_channels=num_filters//4,
                                   kernel_size=kernel_size,
                                   normalization=normalization,
                                   norm_kwargs=norm_kwargs,
                                   conv_padding=conv_padding,
                                   padding_mode=padding_mode,
                                   init=init,
                                   nonlinearity=nonlinearity,
                                   ndim=ndim)]
        self.op += [norm_nlin_conv(in_channels=num_filters//4,
                                   out_channels=num_filters,
                                   kernel_size=1,
                                   upsample=upsample,
                                   upsample_mode=upsample_mode,
                                   normalization=normalization,
                                   norm_kwargs=norm_kwargs,
                                   conv_padding=conv_padding,
                                   padding_mode=padding_mode,
                                   init=init,
                                   nonlinearity=nonlinearity,
                                   ndim=ndim)]
        if dropout > 0:
            self.op += [get_dropout(dropout, nonlinearity)]
        self.op = nn.ModuleList(self.op)
        self.op_shortcut = None
        if skip:
            self.op_shortcut = shortcut(in_channels=in_channels,
                                        out_channels=num_filters,
                                        subsample=subsample,
                                        upsample=upsample,
                                        upsample_mode=upsample_mode,
                                        init=init,
                                        ndim=ndim)

    def forward(self, input):
        out = input
        for op in self.op:
            out = op(out)
        if self.skip:
            out = self.op_shortcut(input, out)
        return out


class conv_block(block_abstract):
    """
    A single basic 3x3 convolution.
    Unlike in tiny_block, stride instead of maxpool and upsample before conv.
    """
    def __init__(self, in_channels, num_filters, subsample=False,
                 upsample=False, upsample_mode='repeat', skip=True, dropout=0.,
                 normalization=batch_normalization, norm_kwargs=None,
                 conv_padding=True, padding_mode='constant', kernel_size=3,
                 init='kaiming_normal_', nonlinearity='ReLU', ndim=2):
        super(conv_block, self).__init__(in_channels, num_filters,
                                         subsample, upsample)
        if norm_kwargs is None:
            norm_kwargs = {}
        self.out_channels = num_filters
        self.upsample_mode = upsample_mode
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        self.op = []
        if normalization is not None:
            self.op += [normalization(ndim=ndim,
                                      num_features=in_channels,
                                      **norm_kwargs)]
        self.op += [get_nonlinearity(nonlinearity)]
        if upsample:
            self.op += [do_upsample(mode=upsample_mode,
                                    ndim=ndim,
                                    in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=2,
                                    init=init)]
        stride = 1
        if subsample:
            stride = 2
        if conv_padding:
            # For odd kernel sizes, equivalent to kernel_size//2.
            # For even kernel sizes, two cases:
            #    (1) [kernel_size//2-1, kernel_size//2] @ stride 1
            #    (2) kernel_size//2 @ stride 2
            # This way, even kernel sizes yield the same output size as
            # odd kernel sizes. When subsampling, even kernel sizes allow
            # possible downscaling without aliasing.
            padding = [(kernel_size-1)//2,
                       (kernel_size-int(subsample))//2]*ndim
        else:
            padding = 0
        self.op += [convolution(in_channels=in_channels,
                                out_channels=num_filters,
                                kernel_size=kernel_size,
                                ndim=ndim,
                                stride=stride,
                                init=init,
                                padding=padding,
                                padding_mode=padding_mode)]
        if dropout > 0:
            self.op += [get_dropout(dropout, nonlin=nonlinearity)]
        self.op = nn.ModuleList(self.op)
        self.op_shortcut = None
        if skip:
            self.op_shortcut = shortcut(in_channels=in_channels,
                                        out_channels=num_filters,
                                        subsample=subsample,
                                        upsample=upsample,
                                        upsample_mode=upsample_mode,
                                        init=init,
                                        ndim=ndim)

    def forward(self, input):
        out = input
        for op in self.op:
            out = op(out)
        if self.skip:
            out = self.op_shortcut(input, out)
        return out


class pool_block(block_abstract):
    """
    A single basic 3x3 convolution.
    """
    def __init__(self, in_channels, num_filters, subsample=False,
                 upsample=False, skip=True, dropout=0.,
                 normalization=batch_normalization, norm_kwargs=None,
                 conv_padding=True, padding_mode='constant', kernel_size=3,
                 upsample_mode='not settable', init='kaiming_normal_',
                 nonlinearity='ReLU', ndim=2):
        super(pool_block, self).__init__(in_channels, num_filters,
                                         subsample, upsample)
        if norm_kwargs is None:
            norm_kwargs = {}
        self.out_channels = num_filters
        self.upsample_mode = 'repeat'   # For `get_output_shape`.
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        self.op = []
        if normalization is not None:
            self.op += [normalization(ndim=ndim,
                                      num_features=in_channels,
                                      **norm_kwargs)]
        self.op += [get_nonlinearity(nonlinearity)]
        if subsample:
            self.op += [max_pooling(kernel_size=2, ndim=ndim,
                                    return_indices=True)]
        if conv_padding:
            # For odd kernel sizes, equivalent to kernel_size//2.
            # For even kernel sizes, two cases:
            #    (1) [kernel_size//2-1, kernel_size//2] @ stride 1
            #    (2) kernel_size//2 @ stride 2
            # This way, even kernel sizes yield the same output size as
            # odd kernel sizes. When subsampling, even kernel sizes allow
            # possible downscaling without aliasing.
            padding = [(kernel_size-1)//2,
                       (kernel_size-int(subsample))//2]*ndim
        else:
            padding = 0
        self.op += [convolution(in_channels=in_channels,
                                out_channels=num_filters,
                                kernel_size=kernel_size,
                                ndim=ndim,
                                init=init,
                                padding=padding,
                                padding_mode=padding_mode)]
        if dropout > 0:
            self.op += [get_dropout(dropout, nonlinearity)]
        if upsample:
            self.op += [max_unpooling(kernel_size=2, ndim=ndim)]
        self.op = nn.ModuleList(self.op)
        self.op_shortcut = None
        if skip:
            self.op_shortcut = shortcut(in_channels=in_channels,
                                        out_channels=num_filters,
                                        subsample=subsample,
                                        upsample=upsample,
                                        upsample_mode=upsample_mode,
                                        init=init,
                                        ndim=ndim)

    def forward(self, input, unpool_indices=None):
        out = input
        indices = None
        for op in self.op:
            if  isinstance(op, (nn.MaxPool2d, nn.MaxPool3d)):
                out, indices = op(out)
            elif unpool_indices is not None \
                         and isinstance(op, (nn.MaxUnpool2d, nn.MaxUnpool3d)):
                out = op(out, unpool_indices)
            else:
                out = op(out)
        if self.skip:
            out = self.op_shortcut(input, out)
        if indices is not None:
            return out, indices
        return out


class AdaptiveInstanceNorm2d(nn.Module):
    """
    From MUNIT.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, ndim=None):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert(self.weight is not None and self.bias is not None,
               "Please assign weight and bias before calling AdaIN!")
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


def grad_norm(module, reduce=False):
    """
    Compute a set of gradient norms, one for every parameter in the module.
    If `reduce` is True, reduce the set to an average value.
    """
    parameters = filter(lambda p: p.grad is not None, module.parameters())
    norm = torch.Tensor([torch.norm(p.grad) for p in parameters])
    if reduce:
        norm = norm.mean()
    return norm


def recursive_spectral_norm(module, types=None):
    """
    Recursively traverse submodules in a module and apply spectral norm to
    all convolutional layers.
    """
    if types is None:
        types = tuple()
    for m in module.modules():
        if isinstance(m, (nn.Conv1d,
                          nn.Conv2d,
                          nn.Conv3d,
                          nn.ConvTranspose1d,
                          nn.ConvTranspose2d,
                          nn.ConvTranspose3d)+types):
            if not hasattr(m, '_has_spectral_norm'):
                spectral_norm(m)
            setattr(m, '_has_spectral_norm', True)


def recursive_remove_spectral_norm(module, types=None):
    """
    Recursively traverse submodules in a module and remove spectral norm from
    all convolutional layers.
    """
    if types is None:
        types = tuple()
    for m in module.modules():
        if isinstance(m, (nn.Conv1d,
                          nn.Conv2d,
                          nn.Conv3d,
                          nn.ConvTranspose1d,
                          nn.ConvTranspose2d,
                          nn.ConvTranspose3d)+types):
            if not hasattr(m, '_has_spectral_norm'):
                remove_spectral_norm(m)
            setattr(m, '_has_spectral_norm', False)
