from __future__ import (print_function,
                        division)
import numpy as np
import torch
from torch.autograd import Variable

def _to_cuda(x, device=None):
    x.data = x.data.cuda(device)
    if x._grad is not None:
        x._grad.data = x._grad.data.cuda(device)
    return x


class _overlap_tile_function(torch.autograd.Function):
    @staticmethod
    def _patch_slice_iterator(input_patch_size, output_patch_size,
                              input_size, overlap):
        lost = np.subtract(input_patch_size, output_patch_size)
        ndim = len(input_patch_size)+2
        stride = np.subtract(output_patch_size, overlap)
        
        #Iterator over indices in an image, for sourcing or placing patches.
        def index_iterator(offset=None):
            if offset is None:
                offset = [0]*(ndim-2)
            def recursive_idx(dim, idx):
                for i in range(0, input_size[dim]+offset[dim-2],
                               stride[dim-2]):
                    idx[dim] = i
                    if dim < ndim-1:
                        for idx in recursive_idx(dim+1, idx):
                            yield idx
                    else:
                        yield idx
            iterator = recursive_idx(dim=2, idx=[None]*ndim)
            return iterator
        
        # Return list of slices for selecting a patch.
        def get_patch_slices(corner_idx, patch_size):
            indices = [slice(None, None)]*ndim
            for dim in range(ndim):
                if corner_idx[dim] is not None:
                    indices[dim] = slice(corner_idx[dim],
                                         corner_idx[dim]+patch_size[dim-2])
            return indices
    
        # Process input patch-wise.
        iter_input = index_iterator(offset=lost//2)
        iter_output = index_iterator()
        for idx_i, idx_o in zip(iter_input, iter_output):
            sl_i = get_patch_slices(idx_i, input_patch_size)
            sl_o = get_patch_slices(idx_o, output_patch_size)
            yield tuple(sl_i), tuple(sl_o)
    
    @staticmethod
    def _create_padded_buffer(input_patch_size, output_patch_size, input_size,
                              overlap, use_gpu=False, device=None):
        ndim = len(input_patch_size)+2
        stride = np.subtract(output_patch_size, overlap)
        
        # Compute the required size of the padded input image.
        lost = np.subtract(input_patch_size, output_patch_size)
        new_shape = list(input_size[:2])
        for i in range(ndim-2):
            pad0 = lost[i]//2
            pad1 = max(lost[i]-pad0,
                       output_patch_size[i]-input_size[i+2]%stride[i]+1)
            padded_length = pad0 + input_size[i+2] + pad1
            new_shape.append(int(padded_length))
                            
        # Create padded buffer.
        input_padded = torch.zeros(*new_shape).float()
        if use_gpu:
            input_padded = input_padded.cuda(device)
            
        # Determine indices of the region bounded by padding.
        indices = [slice(None, None)]*ndim
        spatial_dims = range(2, ndim)
        for dim in spatial_dims:
            pad0 = lost[dim-2]//2
            indices[dim] = slice(pad0, input_size[dim]+pad0)
        
        return input_padded, tuple(indices)
    
    @staticmethod
    def forward(ctx, input, model, input_patch_size, output_patch_size,
                overlap, out_channels, *args):
        ndim = len(input_patch_size)+2
        input_size = input.size()
        use_gpu = True if input.is_cuda else False
        device = None if not input.is_cuda else input.get_device()
        
        # Create and fill padded input image.
        input_padded, indices = _overlap_tile_function._create_padded_buffer(\
                                                        input_patch_size,
                                                        output_patch_size,
                                                        input_size=input_size,
                                                        overlap=overlap,
                                                        use_gpu=use_gpu,
                                                        device=device)
        
        input = input.data
        if use_gpu:
            input = input.cuda(device)
        input_padded[indices] = input
        
        # Create output image buffer.
        output_size = list(input_size)
        output_size[1] = out_channels
        output = torch.zeros(*tuple(output_size)).float()
        if use_gpu:
            output = output.cuda(device)
        
        # Process input patch-wise.
        graph_created = False
        input_patch_list = []
        output_patch_list = []
        iter_type = _overlap_tile_function._patch_slice_iterator
        for sl_i, sl_o in iter_type(input_patch_size, output_patch_size,
                                    input_size, overlap):
            input_patch_data = input_padded[sl_i].contiguous()
            input_patch = Variable(input_patch_data, requires_grad=True)
            if use_gpu:
                input_patch = _to_cuda(input_patch, device)
            if not graph_created:
                with torch.enable_grad():
                    output_patch = model(input_patch)
                    graph_created = True
            else:
                with torch.no_grad():
                    output_patch = model(input_patch)
            sl_p = [slice(None, None)]*2
            for dim in range(2, ndim):
                stop = min(output.size()[dim] - sl_o[dim].start,
                           output_patch.size()[dim])
                sl_p.append(slice(0, stop))
            output[sl_o] += output_patch[sl_p]
            output_patch_list.append(output_patch)
            input_patch_list.append(input_patch)
            
        # Save input, activations, params for backprop.
        ctx.input_patch_list = input_patch_list
        ctx.output_patch_list = output_patch_list
        ctx.save_for_backward(*args)
        
        # Store values for backprop.
        ctx.model = model
        ctx.input_patch_size = input_patch_size
        ctx.output_patch_size = output_patch_size
        ctx.use_gpu = use_gpu
        ctx.device = device
        ctx.input_size = input_size
        ctx.overlap = overlap
        
        del input_padded, input

        #output.data = output.data.contiguous()
        return Variable(output)
    
    @staticmethod
    def backward(ctx, output_grad):
        ndim = len(ctx.input_patch_size)+2
        
        # Create buffer for input gradients.
        input_grad = None
        if ctx.needs_input_grad[0]:
            input_grad = torch.zeros(*tuple(ctx.input_size)).float()
            if ctx.use_gpu:
                input_grad = input_grad.cuda(ctx.device)
                
        # Create buffer for output gradient patches (patches must all be of
        # the same size).
        output_grad_patch_size = tuple(ctx.input_size)[:2] \
                                 + tuple(ctx.output_patch_size)
        _output_grad_patch = torch.zeros(*output_grad_patch_size).float()
        if ctx.use_gpu:
            _output_grad_patch = _output_grad_patch.cuda(ctx.device)
        
        # Get parameters.
        params = ctx.saved_variables
            
        # Process input patch-wise.
        params_grad = [None]*len(params)
        iter_type = _overlap_tile_function._patch_slice_iterator
        _output_patch = None
        _input_patch = None
        for i, (sl_i, sl_o) in enumerate(iter_type(ctx.input_patch_size,
                                                   ctx.output_patch_size,
                                                   ctx.input_size,
                                                   ctx.overlap)):
            input_patch = ctx.input_patch_list[i]
            output_patch = ctx.output_patch_list[i]
            output_grad_patch = output_grad[sl_o]
            sl_g = []
            for s in output_grad_patch.size():
                sl_g.append(slice(0, s))
            _output_grad_patch[sl_g] = output_grad_patch
            
            # Re-use the graph of the first patch.
            if _output_patch is None:
                _output_patch = output_patch
            else:
                _output_patch.data = output_patch.data
            if _input_patch is None:
                _input_patch = input_patch
            else:
                _input_patch.data = _input_patch.data
            assert _output_patch.data.size()==output_patch.data.size()

            # Get gradients
            if input_grad is not None:
                grad_patch = torch.autograd.grad(_output_patch,
                                                 (_input_patch,)+tuple(params),
                                                 _output_grad_patch,
                                                 retain_graph=True)
                input_grad_patch = grad_patch[0]
                params_grad_patch = grad_patch[1:]
            else:
                grad_patch = torch.autograd.grad(_output_patch,
                                                 tuple(params),
                                                 _output_grad_patch,
                                                 retain_graph=True)
                params_grad_patch = grad_patch
            
            # Accumulate gradients.
            if input_grad is not None:
                sl_p = [slice(None, None)]*2
                for dim in range(2, ndim):
                    stop = min(input_grad.size()[dim] - sl_o[dim].start,
                            input_grad.size()[dim])
                    sl_p.append(slice(0, stop))
                input_grad[sl_i] += input_grad_patch.data[sl_p]
            for i, p_grad in enumerate(params_grad_patch):
                if params_grad[i] is None:
                    params_grad[i] = p_grad
                else:
                    params_grad[i] += p_grad
                        
            if input_grad is not None:
                input_grad = Variable(input_grad)
            
        del ctx.input_patch_list[:], ctx.output_patch_list[:]
        
        return (input_grad,) + (None,)*5 + tuple(params_grad)


class overlap_tile(torch.nn.Module):
    """
    Implements the overlap-tile strategy to save memory at the cost of extra
    compute time (both training and inference).
    
    Args:
        input_patch_size (tuple of int) : The spatial size of the overlapping
            tiles to take from the input. The output tile size is determined
            automatically.
        model (Module) : A pytorch module implementing the model/layer to
            do overlap-tile with. Convolutions must have no zero-padding,
            must have stride of 1, and no spatial pooling; the output after
            overlap-tile is assumed to have the same spatial size as the input.
        in_channels (int) : The number of input channels.
        out_channels (int) : The number of output channels.
        overlap (tuple of ints) : For each spatial dimension, the amount by
            which output patches should be made to overlap.
        verbose (bool) : Whether to print the autodetermined output patch
            size.
    """
    def __init__(self, input_patch_size, model, in_channels, out_channels,
                 overlap=None, verbose=True):
        super(overlap_tile, self).__init__()
        self.input_patch_size = input_patch_size
        self.model = model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_patch_size = None
        self.verbose = verbose
        self.ndim = len(self.input_patch_size)
        self.overlap = overlap
        if self.overlap is None:
            self.overlap = (0,)*self.ndim
        
        
    def parameters(self):
        return self.model.parameters()
    
    def named_parameters(self, memo=None, prefix=''):
        return self.model.named_parameters()
        
    def forward(self, input):
        if input.ndimension() != self.ndim+2:
            raise ValueError("Using an input patch size of {}, expecting the "
                             "input to have {} dimensions. Got {} dimensions."
                             "".format(self.input_patch_size, self.ndim+2,
                                       input.ndimension()))
                                       
        # If not already computed, compute the output patch size.
        # HACK: this uses a forward pass.
        if self.output_patch_size is None:
            input_size = (1,self.in_channels)+tuple(self.input_patch_size)
            _in = Variable(torch.zeros(*input_size).float())
            if input.is_cuda:
                _in = _to_cuda(_in, input.get_device())
            _out = self.model(_in)
            self.output_patch_size = tuple(_out.size()[2:])
            if self.verbose:
                print("OVERLAP-TILE: For input patch size {}, output patch "
                      "size is {} (overlap: {})."
                      "".format(self.input_patch_size, self.output_patch_size,
                                self.overlap))
            
        # Forward pass.
        output = _overlap_tile_function.apply(input,
                                              self.model,
                                              self.input_patch_size,
                                              self.output_patch_size,
                                              self.overlap,
                                              self.out_channels,
                                              *self.model.parameters())
            
        return output
 

if __name__=='__main__':
    class identity_module(torch.nn.Module):
        def forward(self, input):
            return input

    #model = identity_module()
    model = torch.nn.Sequential( \
                torch.nn.Conv2d(1, 1, 3, padding=1),
                torch.nn.Conv2d(1, 1, 3, dilation=2, padding=2),
                torch.nn.Conv2d(1, 1, 3, dilation=4, padding=4))
    model.cuda()
    tiler = overlap_tile(input_patch_size=(10,9), model=model,
                         in_channels=1, out_channels=1)
    tiler.cuda()

    edge = 100
    im_input = np.reshape(range(edge**2), (1,1,edge,edge)).astype(np.float32)
    im_input = Variable(torch.from_numpy(im_input))
    im_input = im_input.cuda()
    im_output = model(im_input)
    im_output_tiled = tiler(im_input)

    if np.all(np.abs((im_output-im_output_tiled).data) < 0.0001):
        print("Test PASSED")
    else:
        print("Test FAILED")
    #print(im_input.size(), im_output.size())
        
    #loss = im_output.mean()-1
    #loss.backward()
