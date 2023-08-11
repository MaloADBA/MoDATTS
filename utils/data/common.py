import os
import imageio
import numpy as np
from skimage import transform

from data_tools.wrap import delayed_view
from data_tools.io import data_flow


def resize_stack(arr, size, interp='bilinear'):
    """
    Resize each slice of a tensor, assuming the last two dimensions span the
    slice.
    """
    out = np.zeros(arr.shape[:-2]+size, dtype=arr.dtype)
    for idx in np.ndindex(arr.shape[:-2]):
        out[idx] = transform.resize(arr[idx],
                                    output_shape=size,
                                    mode='constant',
                                    cval=0,
                                    clip=True,
                                    preserve_range=True)
    return out


class folder_indexer(object):
    """
    Single-value index interface for loading images from a folder.
    """
    def __init__(self, path):
        self.path = path
        self.fn_list = sorted(os.listdir(path))
        self.shape = (len(self.fn_list),)
        self.dtype = 'strings'
      
    def __getitem__(self, index):
        fn = self.fn_list[index]
        return imageio.imread(os.path.join(self.path, fn))
    
    def __len__(self):
        return len(self.fn_list)


class data_flow_sampler(data_flow):
    """
    A data_flow implementation that does not restrict every source in `data`
    to be of equal length. 
    
    Whenever an iterator is created by data_flow (eg. each epoch), a random
    (equal length) non-contiguous subset of data is viewed from each source
    that is longer than the shortest length.
    
    NOTE: this means that these sources are shuffled, regardless of which
          sampling arguments are passed.
    """
    def __init__(self, data, *args, **kwargs):
        min_length = min(len(d) for d in data)
        _data = []
        self.reshuffle_indicator = []
        for d in data:
            if len(d) > min_length:
                d = delayed_view(d, shuffle=True)
                d = delayed_view(d, idx_min=0, idx_max=min_length)
                self.reshuffle_indicator.append(True)
            else:
                self.reshuffle_indicator.append(False)
            _data.append(d)
            
        super(data_flow_sampler, self).__init__(data=_data, *args, **kwargs)
        
    def flow(self):
        for i, reshuffle in enumerate(self.reshuffle_indicator):
            if reshuffle:
                self.data[i].arr.re_shuffle()
        return super(data_flow_sampler, self).flow()
        

class masked_view(delayed_view):
    """
    Given an array, mask some random subset of indices, returning `None`
    for at these indices.
    
    arr : The source array.
    masked_fraction : Fraction of elements to mask (length axis), in [0, 1.0].
    rng : A numpy random number generator.
    """
    
    def __init__(self, arr, masked_fraction, rng=None):
        super(masked_view, self).__init__(arr=arr, shuffle=False, rng=rng)
        if 0 > masked_fraction or masked_fraction > 1:
            raise ValueError("In `masked_view`, `masked_fraction` must be set "
                             "to a value in [0, 1.0] but was set to {}."
                             "".format(masked_fraction))
        self.masked_fraction = masked_fraction
        num_masked_indices = int(min(self.num_items,
                                     masked_fraction*self.num_items+0.5))
        self.masked_indices = self.rng.choice(self.num_items,
                                              size=num_masked_indices,
                                              replace=False)
        
    def _get_element(self, int_key, key_remainder=None):
        if not isinstance(int_key, (int, np.integer)):
            raise IndexError("cannot index with {}".format(type(int_key)))
        if int_key in self.masked_indices:
            return None
        idx = self.arr_indices[int_key]
        if key_remainder is not None:
            idx = (idx,)+key_remainder
        idx = int(idx)  # Some libraries don't like np.integer
        return self.arr[idx]


class permuted_view(delayed_view):
    """
    Given an array, permute some random subset of indices, making their
    mapping intentionally incorrect.
    
    arr : The source array.
    fraction : Fraction of elements to permute (length axis), in [0, 1.0].
    rng : A numpy random number generator.
    """
    
    def __init__(self, arr, fraction, rng=None):
        super().__init__(arr=arr, shuffle=False, rng=rng)
        if 0 > fraction or fraction > 1:
            raise ValueError("In `permuted_view`, `fraction` must be set "
                             "to a value in [0, 1.0] but was set to {}."
                             "".format(fraction))
        self.fraction = fraction
        n = int(min(self.num_items, fraction*self.num_items+0.5))
        permuted_indices = self.rng.choice(self.num_items,
                                           size=n,
                                           replace=False)
        self.remap = dict(zip(permuted_indices, permuted_indices[::-1]))
        
    def _get_element(self, int_key, key_remainder=None):
        if not isinstance(int_key, (int, np.integer)):
            raise IndexError("cannot index with {}".format(type(int_key)))
        if int_key in self.remap:
            int_key = self.remap[int_key]
        idx = self.arr_indices[int_key]
        if key_remainder is not None:
            idx = (idx,)+key_remainder
        idx = int(idx)  # Some libraries don't like np.integer
        return self.arr[idx]
