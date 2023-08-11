from collections import OrderedDict
import torch
from torch import nn
from torch.nn.utils import remove_spectral_norm
from torch.functional import F
from torch.nn.utils import spectral_norm
import numpy as np
from fcn_maker.model import assemble_resunet
from fcn_maker.loss import dice_loss
from model.common.network.basic import (AdaptiveInstanceNorm2d,
                                        adjust_to_size,
                                        batch_normalization,
                                        basic_block,
                                        convolution,
                                        conv_block,
                                        do_upsample,
                                        get_initializer,
                                        get_nonlinearity,
                                        get_output_shape,
                                        instance_normalization,
                                        layer_normalization,
                                        munit_discriminator,
                                        norm_nlin_conv,
                                        pool_block,
                                        recursive_spectral_norm,
                                        repeat_block)
from model.common.losses import dist_ratio_mse_abs
from model.gen_3d_self_training_mbrats import segmentation_model
from einops import rearrange

def build_model(lambda_disc=1,
                lambda_x_id=1,
                lambda_z_id=1,
                lambda_seg=1, 
                lambda_enforce_sum=1):
    
    
    encoder = Encoder(in_chan=1, num_classes=1, base_chan=32, map_size=[3,10,4], 
                conv_block='BasicBlock', conv_num=[2,0,0,0, 0,0,2,2], 
                trans_num=[0,2,4,6, 4,2,0,0], chan_num=[64,128,256,320,256,128,64,32], 
                num_heads=[1,4,8,10, 8,4,1,1], fusion_depth=2, fusion_dim=320, 
                fusion_heads=10, expansion=4, attn_drop=0., proj_drop=0., proj_type='depthwise', 
                norm='in', act='gelu', kernel_size=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]], 
                scale=[[2,2,2],[2,2,2],[2,2,2],[2,2,2]], aux_loss=False)
    
    decoder = Decoder(in_chan=1, num_classes=1, base_chan=32, map_size=[3,10,4], 
                conv_block='BasicBlock', conv_num=[2,0,0,0, 0,0,2,2], 
                trans_num=[0,2,4,6, 4,2,0,0], chan_num=[64,128,256,320,256,128,64,32], 
                num_heads=[1,4,8,10, 8,4,1,1], fusion_depth=2, fusion_dim=320, 
                fusion_heads=10, expansion=4, attn_drop=0., proj_drop=0., proj_type='depthwise', 
                norm='in', act='gelu', kernel_size=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]], 
                scale=[[2,2,2],[2,2,2],[2,2,2],[2,2,2]], aux_loss=False)
    
    submodel = {
        'encoder'           : encoder,
        'decoder_common'    : None,
        'decoder_residual'  : decoder,  
        'segmenter'         : None,
        'disc_TA'            : None,
        'disc_TB'            : None}
    
    
    model = segmentation_model(**submodel,
                               shape_sample=None,
                               loss_gan='hinge',
                               loss_seg=dice_loss([1]),
                               relativistic=False,
                               rng=np.random.RandomState(1234),
                               lambda_seg=1)
    
    return {'G' : model}


class Encoder(nn.Module):
    
    def __init__(self, in_chan, num_classes, base_chan=32, map_size=[4,8,8], 
        conv_block='BasicBlock', conv_num=[2,1,0,0, 0,1,2,2], trans_num=[0,1,2,2, 2,1,0,0], 
        chan_num=[64,128,256,320,256,128,64,32], num_heads=[1,4,8,16, 8,4,1,1], fusion_depth=2, 
        fusion_dim=320, fusion_heads=4, expansion=4, attn_drop=0., proj_drop=0., 
        proj_type='depthwise', norm='in', act='gelu', kernel_size=[3,3,3,3], scale=[2,2,2,2], aux_loss=False):
        super().__init__()

        if conv_block == 'BasicBlock':
            dim_head = [chan_num[i]//num_heads[i] for i in range(8)]

        
        conv_block = get_block(conv_block)
        norm = get_norm(norm)
        act = get_act(act)
        
        # self.inc and self.down1 forms the conv stem
        self.inc = inconv(in_chan, base_chan, block=conv_block, kernel_size=kernel_size[0], norm=norm, act=act)
        self.down1 = down_block(base_chan, chan_num[0], conv_num[0], trans_num[0], conv_block=conv_block, kernel_size=kernel_size[1], down_scale=scale[0], norm=norm, act=act, map_generate=False)
        
        # down2 down3 down4 apply the B-MHA blocks
        self.down2 = down_block(chan_num[0], chan_num[1], conv_num[1], trans_num[1], conv_block=conv_block, kernel_size=kernel_size[2], down_scale=scale[1], heads=num_heads[1], dim_head=dim_head[1], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)

        self.down3 = down_block(chan_num[1], chan_num[2], conv_num[2], trans_num[2], conv_block=conv_block, kernel_size=kernel_size[3], down_scale=scale[2], heads=num_heads[2], dim_head=dim_head[2], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)

        self.down4 = down_block(chan_num[2], chan_num[3], conv_num[3], trans_num[3], conv_block=conv_block, kernel_size=kernel_size[4], down_scale=scale[3], heads=num_heads[3], dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
   
    def forward(self, x):
       
        x0 = self.inc(x)
        x1, _ = self.down1(x0)
        x2, map2 = self.down2(x1)
        x3, map3 = self.down3(x2)
        x4, map4 = self.down4(x3)
        
        
        
        map_list = [map2, map3, map4]

        return x4, [x0, x1, x2, x3], map_list

    
class Decoder(nn.Module):
    
    def __init__(self, in_chan, num_classes, base_chan=32, map_size=[4,8,8], 
        conv_block='BasicBlock', conv_num=[2,1,0,0, 0,1,2,2], trans_num=[0,1,2,2, 2,1,0,0], 
        chan_num=[64,128,256,320,256,128,64,32], num_heads=[1,4,8,16, 8,4,1,1], fusion_depth=2, 
        fusion_dim=320, fusion_heads=4, expansion=4, attn_drop=0., proj_drop=0., 
        proj_type='depthwise', norm='in', act='gelu', kernel_size=[3,3,3,3], scale=[2,2,2,2], aux_loss=False):
        super().__init__()

        if conv_block == 'BasicBlock':
            dim_head = [chan_num[i]//num_heads[i] for i in range(8)]

        
        conv_block = get_block(conv_block)
        norm = get_norm(norm)
        act = get_act(act)
        
        self.map_fusion = SemanticMapFusion(chan_num[1:4], fusion_dim, fusion_heads, depth=fusion_depth, norm=norm)

        self.up1 = up_block(chan_num[3], chan_num[4], conv_num[4], trans_num[4], conv_block=conv_block, kernel_size=kernel_size[3], up_scale=scale[3], heads=num_heads[4], dim_head=dim_head[4], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)

        self.up2 = up_block(chan_num[4], chan_num[5], conv_num[5], trans_num[5], conv_block=conv_block, kernel_size=kernel_size[2], up_scale=scale[2], heads=num_heads[5], dim_head=dim_head[5], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)

        self.up3 = up_block(chan_num[5], chan_num[6], conv_num[6], trans_num[6], conv_block=conv_block, kernel_size=kernel_size[1], up_scale=scale[1], norm=norm, act=act, map_shortcut=False)

        self.up4 = up_block(chan_num[6], chan_num[7], conv_num[7], trans_num[7], conv_block=conv_block, kernel_size=kernel_size[0], up_scale=scale[0], norm=norm, act=act, map_shortcut=False)

        self.outc = nn.Conv3d(chan_num[7], num_classes, kernel_size=1)

    def forward(self, semantic_m, skips, map_l):
       
        map_list = self.map_fusion(map_l)
        
        out, semantic_map = self.up1(semantic_m, skips[3], map_list[2], map_list[1])
        out, semantic_map = self.up2(out, skips[2], semantic_map, map_list[0])

        out, semantic_map = self.up3(out, skips[1], semantic_map, None)
        out, semantic_map = self.up4(out, skips[0], semantic_map, None)
    
        out = self.outc(out)

        return out
    
    
class Mlp(nn.Module):
    def __init__(self, in_dim, hid_dim=None, out_dim=None, act=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hid_dim = hid_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = act()
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x): 
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: B, L, C.   Batch, sequence length, dim
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b l (heads dim_head) -> b heads l dim_head', heads=self.heads), [q, k, v])
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = F.softmax(attn, dim=-1)

        attned = torch.einsum('bhij,bhjd->bhid', attn, v)
        attned = rearrange(attned, 'b heads l dim_head -> b l (dim_head heads)')

        attned = self.to_out(attned)

        return attned


class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, attn_drop, proj_drop)),
                PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))
                ]))
    def forward(self, x):
        
        for attn, ffn in self.layers:
            x = attn(x) + x
            x = ffn(x) + x

        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):

        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[None, :, None, None, None] * x + self.bias[None, :, None, None, None]
            return x
        
class ConvNormAct(nn.Module):
    """
    Layer grouping a convolution, normalization and activation function
    normalization includes BN as IN
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
        groups=1, dilation=1, bias=False, norm=nn.BatchNorm3d, act=nn.ReLU, preact=False):

        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv = nn.Conv3d(
            in_channels=in_ch, 
            out_channels=out_ch, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias
        )
        if preact:
            self.norm = norm(in_ch) if norm else nn.Identity()
        else:
            self.norm = norm(out_ch) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
        self.preact = preact

    def forward(self, x): 
    
        if self.preact:
            out = self.conv(self.act(self.norm(x)))
        else:
            out = self.act(self.norm(self.conv(x)))

        return out 


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=False):
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i//2 for i in kernel_size]

        self.conv = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)

    def forward(self, x):

        return self.conv(x)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i//2 for i in kernel_size]

        self.conv1 = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch, out_ch, kernel_size, stride=1, padding=pad_size, norm=norm, act=act, preact=preact)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += self.shortcut(residual)

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, groups=1, dilation=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i//2 for i in kernel_size]

        self.expansion = 2
        self.conv1 = ConvNormAct(in_ch, out_ch//self.expansion, 1, stride=1, padding=0, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch//self.expansion, out_ch//self.expansion, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, groups=groups, dilation=dilation, preact=preact)

        self.conv3 = ConvNormAct(out_ch//self.expansion, out_ch, 1, stride=1, padding=0, norm=norm, act=act, preact=preact)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += self.shortcut(residual)

        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, bias=False):
        super().__init__()
        
        if isinstance(kernel_size, list):
            padding = [i//2 for i in kernel_size]
        else:
            padding = kernel_size // 2

        self.depthwise = nn.Conv3d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=bias
        )
        self.pointwise = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias
        )
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out

class SEBlock(nn.Module):
    def __init__(self, in_ch, ratio=4, act=nn.ReLU):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
                        nn.Conv3d(in_ch, in_ch//ratio, kernel_size=1),
                        act(),
                        nn.Conv3d(in_ch//ratio, in_ch, kernel_size=1),
                        nn.Sigmoid()
        )
    def forward(self, x):
        out = self.squeeze(x)
        out = self.excitation(out)

        return x * out


class DropPath(nn.Module):
    
    def __init__(self, p=0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if (not self.p) or (not self.training):
            return x

        batch_size = x.shape[0]
        random_tensor = torch.rand(batch_size, 1, 1, 1, 1).to(x.device)
        binary_mask = self.p < random_tensor

        x = x.div(1 - self.p)
        x = x * binary_mask

        return x


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, stride=1, ratio=4, p=0, se=True, norm=nn.BatchNorm3d, act=nn.ReLU):
        super().__init__()

        if isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2
        else:
            padding = [(t-1)//2 for t in kernel_size]
        expanded = expansion * in_ch
        self.se = se

        self.expand_proj = nn.Identity() if (expansion==1) else ConvNormAct(in_ch, expanded, kernel_size=1, padding=0, norm=norm, act=act, preact=True)

        self.depthwise = ConvNormAct(expanded, expanded, kernel_size=kernel_size, stride=stride, padding=padding, groups=expanded, act=act, norm=norm, preact=True)

        if self.se:
            self.se = SEBlock(expanded, ratio=ratio)

        self.pointwise = ConvNormAct(expanded, out_ch, kernel_size=1, padding=0, norm=norm, act=False, preact=True)

        self.drop_path = DropPath(p)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch or stride !=1:
            self.shortcut = nn.Sequential(ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=padding, norm=False, act=False))


    def forward(self, x):
        residual = x

        x = self.expand_proj(x)
        x = self.depthwise(x)
        if self.se:
            x = self.se(x)

        x = self.pointwise(x)

        x = self.drop_path(x)

        x += self.shortcut(residual)

        return x


class FusedMBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, stride=1, ratio=4, p=0, se=True, norm=nn.BatchNorm3d, act=nn.ReLU):
        super().__init__()

        if isinstance(kernel_size, int):
            padding = (kernel_size -1) // 2
        else:
            padding = [(t-1)//2 for t in kernel_size]

        expanded = expansion * in_ch

        self.stride= stride
        self.se = se

        self.conv3x3 = ConvNormAct(in_ch, expanded, kernel_size=kernel_size, stride=stride, padding=padding, groups=1, norm=norm, act=act, preact=True)

        if self.se:
            self.se_block = SEBlock(expanded, ratio=ratio)

        self.pointwise = ConvNormAct(expanded, out_ch, kernel_size=1, padding=0, norm=norm, act=False, preact=True)

        self.drop_path = DropPath(p)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch or stride !=1:
            self.shortcut = nn.Sequential(ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=padding, norm=False, act=False))

    def forward(self, x):
        residual = x

        x = self.conv3x3(x)
        if self.se:
            x = self.se_block(x)

        x = self.pointwise(x)

        x = self.drop_path(x)

        x = x + self.shortcut(residual)

        return x  
    
    
class BidirectionAttention(nn.Module):
    def __init__(self, feat_dim, map_dim, out_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0.,
                    map_size=[8,8,8], proj_type='depthwose', kernel_size=[3,3,3]):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.feat_dim = feat_dim
        self.map_dim = map_dim
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.map_size = map_size
        
        assert proj_type in ['linear', 'depthwise']

        if proj_type == 'linear':
            self.feat_qv = nn.Conv3d(feat_dim, self.inner_dim*2, kernel_size=1, stride=1, padding=0, bias=False)
            self.feat_out = nn.Conv3d(self.inner_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.feat_qv = DepthwiseSeparableConv(feat_dim, self.inner_dim*2, kernel_size=kernel_size)
            self.feat_out = DepthwiseSeparableConv(self.inner_dim, out_dim, kernel_size=kernel_size)

        
        self.map_qv = nn.Conv3d(map_dim, self.inner_dim*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.map_out = nn.Conv3d(self.inner_dim, map_dim, kernel_size=1, stride=1, padding=0, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        
    def forward(self, feat, semantic_map):

        B, C, D, H, W = feat.shape

        feat_q, feat_v = self.feat_qv(feat).chunk(2, dim=1) # B, inner_dim, D, H, W
        map_q, map_v = self.map_qv(semantic_map).chunk(2, dim=1) # B, inner_dim, ms, ms, ms

        feat_q, feat_v = map(lambda t: rearrange(t, 'b (dim_head heads) d h w -> b heads (d h w) dim_head', dim_head=self.dim_head, heads=self.heads, d=D, h=H, w=W, b=B), [feat_q, feat_v])
        map_q, map_v = map(lambda t: rearrange(t, 'b (dim_head heads) d h w -> b heads (d h w) dim_head', dim_head=self.dim_head, heads=self.heads, d=self.map_size[0], h=self.map_size[1], w = self.map_size[2], b=B), [map_q, map_v])


        attn = torch.einsum('bhid,bhjd->bhij', feat_q, map_q)
        attn *= self.scale
        
        feat_map_attn = F.softmax(attn, dim=-1)  # semantic map is very concise that don't need dropout, 
                                                 # add dropout might cause unstable during training
        map_feat_attn = self.attn_drop(F.softmax(attn, dim=-2))

        feat_out = torch.einsum('bhij,bhjd->bhid', feat_map_attn, map_v)
        feat_out = rearrange(feat_out, 'b heads (d h w) dim_head -> b (dim_head heads) d h w', d=D, h=H, w=W)

        map_out = torch.einsum('bhji,bhjd->bhid', map_feat_attn, feat_v)
        map_out = rearrange(map_out, 'b heads (d h w) dim_head -> b (dim_head heads) d h w', d=self.map_size[0], h=self.map_size[1], w=self.map_size[2])


        feat_out = self.proj_drop(self.feat_out(feat_out))
        map_out = self.map_out(map_out)

        return feat_out, map_out




class BidirectionAttentionBlock(nn.Module):
    def __init__(self, feat_dim, map_dim, out_dim, heads, dim_head, norm=nn.BatchNorm3d, act=nn.ReLU,
                expansion=4, attn_drop=0., proj_drop=0., map_size=[8, 8, 8], 
                proj_type='depthwise', kernel_size=[3,3,3]):
        super().__init__()

        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]
        assert proj_type in ['linear', 'depthwise']

        self.norm1 = norm(feat_dim) if norm else nn.Identity() # norm layer for feature map
        self.norm2 = norm(map_dim) if norm else nn.Identity() # norm layer for semantic map
        
        self.attn = BidirectionAttention(feat_dim, map_dim, out_dim, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, kernel_size=kernel_size)

        self.shortcut = nn.Sequential()
        if feat_dim != out_dim:
            self.shortcut = ConvNormAct(feat_dim, out_dim, 1, padding=0, norm=norm, act=act, preact=True)

        if proj_type == 'linear':
            self.feedforward = FusedMBConv(out_dim, out_dim, expansion=expansion, kernel_size=1, act=act, norm=norm)
        else:
            self.feedforward = MBConv(out_dim, out_dim, expansion=expansion, kernel_size=kernel_size, act=act, norm=norm)

    def forward(self, x, semantic_map):
        
        feat = self.norm1(x)
        mapp = self.norm2(semantic_map)

        out, mapp = self.attn(feat, mapp)

        out += self.shortcut(x)
        out = self.feedforward(out)

        mapp += semantic_map

        return out, mapp

class PatchMerging(nn.Module):
    """
    Modified patch merging layer that works as down-sampling
    """
    def __init__(self, dim, out_dim, norm=nn.BatchNorm3d, proj_type='linear', down_scale=[2,2,2], kernel_size=[3,3,3]):
        super().__init__()
        self.dim = dim
        assert proj_type in ['linear', 'depthwise']

        self.down_scale = down_scale

        merged_dim = 2 ** down_scale.count(2) * dim

        if proj_type == 'linear':
            self.reduction = nn.Conv3d(merged_dim, out_dim, kernel_size=1, bias=False)
        else:
            self.reduction = DepthwiseSeparableConv(merged_dim, out_dim, kernel_size=kernel_size)

        self.norm = norm(merged_dim)

    def forward(self, x):
        """
        x: B, C, D, H, W
        """
        merged_x = []
        for i in range(self.down_scale[0]):
            for j in range(self.down_scale[1]):
                for k in range(self.down_scale[2]):
                    tmp_x = x[:, :, i::self.down_scale[0], j::self.down_scale[1], k::self.down_scale[2]]
                    merged_x.append(tmp_x)
        
        x = torch.cat(merged_x, 1)
        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    """
    A basic transformer layer for one stage
    No downsample or upsample operation in this layer, they are wrapped in the down_block of up_block
    """

    def __init__(self, feat_dim, map_dim, out_dim, num_blocks, heads=4, dim_head=64, expansion=4, attn_drop=0., proj_drop=0., map_size=[8,8,8], proj_type='depthwise', norm=nn.BatchNorm3d, act=nn.GELU, kernel_size=[3,3,3]):
        super().__init__()

        dim1 = feat_dim
        dim2 = out_dim

        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            self.blocks.append(BidirectionAttentionBlock(dim1, map_dim, dim2, heads, dim_head, expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, kernel_size=kernel_size))
            dim1 = out_dim

    def forward(self, x, semantic_map):
        for block in self.blocks:
            x, semantic_map = block(x, semantic_map)

        
        return x, semantic_map


class SemanticMapGeneration(nn.Module):
    def __init__(self, feat_dim, map_dim, map_size):
        super().__init__()

        self.map_size = map_size
        self.map_dim = map_dim

        self.map_code_num = map_size[0] * map_size[1] * map_size[2]

        self.base_proj = nn.Conv3d(feat_dim, map_dim, kernel_size=3, padding=1, bias=False)

        self.semantic_proj = nn.Conv3d(feat_dim, self.map_code_num, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        B, C, D, H, W = x.shape

        feat = self.base_proj(x) #B, map_dim, d, h, w
        weight_map = self.semantic_proj(x) # B, map_code_num, d, h, w

        weight_map = weight_map.view(B, self.map_code_num, -1)
        weight_map = F.softmax(weight_map, dim=2) #B, map_code_num, dhw)
        feat = feat.view(B, self.map_dim, -1) # B, map_dim, dhw
        
        semantic_map = torch.einsum('bij,bkj->bik', feat, weight_map)

        return semantic_map.view(B, self.map_dim, self.map_size[0], self.map_size[1], self.map_size[2])


class SemanticMapFusion(nn.Module):
    def __init__(self, in_dim_list, dim, heads, depth=1, norm=nn.BatchNorm3d, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim

        # project all maps to the same channel num
        self.in_proj = nn.ModuleList([])
        for i in range(len(in_dim_list)):
            self.in_proj.append(nn.Conv3d(in_dim_list[i], dim, kernel_size=1, bias=False))

        self.fusion = TransformerBlock(dim, depth, heads, dim//heads, dim, attn_drop=attn_drop, proj_drop=proj_drop)

        # project all maps back to their origin channel num
        self.out_proj = nn.ModuleList([])
        for i in range(len(in_dim_list)):
            self.out_proj.append(nn.Conv3d(dim, in_dim_list[i], kernel_size=1, bias=False))

    def forward(self, map_list):
        B, _, D, H, W = map_list[0].shape
        proj_maps = [self.in_proj[i](map_list[i]).view(B, self.dim, -1).permute(0, 2, 1) for i in range(len(map_list))]
        #B, L, C where L=DHW

        proj_maps = torch.cat(proj_maps, dim=1)
        attned_maps = self.fusion(proj_maps)

        attned_maps = attned_maps.chunk(len(map_list), dim=1)

        maps_out = [self.out_proj[i](attned_maps[i].permute(0, 2, 1).view(B, self.dim, D, H, W)) for i in range(len(map_list))]

        return maps_out
################################################################

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], block=BasicBlock, norm=nn.BatchNorm3d, act=nn.GELU):
        super().__init__()

        pad_size = [i//2 for i in kernel_size]
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=pad_size, bias=False)

        self.conv2 = block(out_ch, out_ch, kernel_size=kernel_size, norm=norm, act=act)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out



class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, conv_num, trans_num, down_scale=[2,2,2], kernel_size=[3,3,3],
                conv_block=BasicBlock, heads=4, dim_head=64, expansion=1, attn_drop=0., 
                proj_drop=0., map_size=[8,8,8], proj_type='depthwise', norm=nn.BatchNorm3d, 
                act=nn.GELU, map_generate=False, map_dim=None):
        super().__init__()
        

        map_dim = out_ch if map_dim is None else map_dim
        self.map_generate = map_generate
        if map_generate:
            self.map_gen = SemanticMapGeneration(out_ch, map_dim, map_size)

        self.patch_merging = PatchMerging(in_ch, out_ch, norm=norm, proj_type=proj_type, down_scale=down_scale, kernel_size=kernel_size)

        block_list = []
        for i in range(conv_num):
            block_list.append(conv_block(out_ch, out_ch, norm=norm, act=act, kernel_size=kernel_size))
        self.conv_blocks = nn.Sequential(*block_list)

        self.trans_blocks = BasicLayer(out_ch, map_dim, out_ch, num_blocks=trans_num, heads=heads, \
                dim_head=dim_head, norm=norm, act=act, expansion=expansion, attn_drop=attn_drop, \
                proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, kernel_size=kernel_size)

    def forward(self, x):
        x = self.patch_merging(x)

        out = self.conv_blocks(x)

        if self.map_generate:
            semantic_map = self.map_gen(out)
        else:
            semantic_map = None

        out, semantic_map = self.trans_blocks(out, semantic_map)
            

        return out, semantic_map

class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, conv_num, trans_num, up_scale=[2,2,2], kernel_size=[3,3,3], 
                conv_block=BasicBlock, heads=4, dim_head=64, expansion=4, attn_drop=0., proj_drop=0.,
                map_size=[4,8,8], proj_type='depthwise', norm=nn.BatchNorm3d, act=nn.GELU, 
                map_dim=None, map_shortcut=False):
        super().__init__()


        self.map_shortcut = map_shortcut
        map_dim = out_ch if map_dim is None else map_dim
        if map_shortcut:
            self.map_reduction = nn.Conv3d(in_ch+out_ch, map_dim, kernel_size=1, bias=False)
        else:
            self.map_reduction = nn.Conv3d(in_ch, map_dim, kernel_size=1, bias=False)

        self.trans_blocks = BasicLayer(in_ch+out_ch, map_dim, out_ch, num_blocks=trans_num, \

                    heads=heads, dim_head=dim_head, norm=norm, act=act, expansion=expansion, \
                    attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size,\
                    proj_type=proj_type, kernel_size=kernel_size)

        if trans_num == 0:
            dim1 = in_ch+out_ch
        else:
            dim1 = out_ch

        conv_list = []
        for i in range(conv_num):
            conv_list.append(conv_block(dim1, out_ch, kernel_size=kernel_size, norm=norm, act=act))
            dim1 = out_ch
        self.conv_blocks = nn.Sequential(*conv_list)

    def forward(self, x1, x2, map1, map2=None):
        # x1: low-res feature, x2: high-res feature shortcut from encoder
        # map1: semantic map from previous low-res layer
        # map2: semantic map from encoder shortcut, might be none if we don't have the map from encoder
        
        x1 = F.interpolate(x1, size=x2.shape[-3:], mode='trilinear', align_corners=True)
        feat = torch.cat([x1, x2], dim=1)
        
        if self.map_shortcut and map2 is not None:
            semantic_map = torch.cat([map1, map2], dim=1)
        else:
            semantic_map = map1
        
        if semantic_map is not None:
            semantic_map = self.map_reduction(semantic_map)

        out, semantic_map = self.trans_blocks(feat, semantic_map)
        out = self.conv_blocks(out)

        return out, semantic_map       
    
def get_block(name):
    block_map = {
        'SingleConv': SingleConv,
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck,
    }
    return block_map[name]

def get_norm(name):
    norm_map = {'bn': nn.BatchNorm3d,
                'in': nn.InstanceNorm3d,
                'ln': LayerNorm
                }

    return norm_map[name]

def get_act(name):
    act_map = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'swish': nn.SiLU
    }
    return act_map[name]