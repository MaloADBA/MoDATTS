from collections import OrderedDict

import h5py
import numpy as np
from scipy import ndimage
import SimpleITK as sitk

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms \
    import (BrightnessMultiplicativeTransform,
            ContrastAugmentationTransform,
            BrightnessTransform)
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms \
    import (GaussianNoiseTransform,
            GaussianBlurTransform)
from batchgenerators.transforms.resample_transforms \
    import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2

from data_tools.data_augmentation import image_random_transform
from data_tools.wrap import multi_source_array

def prepare_mbrats_trans_2d(path, modalities = ['t1', 't2'],
                   drop_masked=False, rng=None):
    
    # Random 20% data split (20% validation).
    rnd_state = np.random.RandomState(0)
    indices = np.arange(0, 369)
    rnd_state.shuffle(indices)
    val  = indices[0:37]
    
    return _prepare_mbrats_trans_2d(path,
                              modalities,
                              validation_indices=val,
                              drop_masked=drop_masked,
                              rng=rng)

def _prepare_mbrats_trans_2d(path, modalities, validation_indices,
                             drop_masked=False, rng=None):
    
    if rng is None:
        rng = np.random.RandomState(0)
    
    # Assemble volumes and corresponding segmentations; split train/valid/test.                                        
    volumes_S = {'train': [], 'valid': []}
    seg_S = {'train': [], 'valid': []}
    volumes_T = {'train': [], 'valid': []}
    
    h5py_file = h5py.File(path, mode='r')
    
    indices = []
    case_identities = []
    
    for idx, case_id in enumerate(h5py_file.keys()):   # Per patient.
        f = h5py_file[case_id]
        if idx in validation_indices:
            split = 'valid'
            volumes_S[split].append(f[modalities[0]])
            volumes_T[split].append(f[modalities[1]])
            seg_S[split].append(f['segmentation'])
        else:
            split = 'train'
            volumes_S[split].append(f[modalities[0]])
            volumes_T[split].append(f[modalities[1]])
            seg_S[split].append(f['segmentation'])
    
    # Merge all arrays in each list of arrays.
    data = OrderedDict([('train', OrderedDict()),
                        ('valid', OrderedDict())])

                                                    
    for key in data.keys():      
        data[key]['source']  = multi_source_array(volumes_S[key])
        data[key]['seg_source']  = multi_source_array(seg_S[key])
        data[key]['target']  = multi_source_array(volumes_T[key])
    return data



def preprocessor_mbrats_trans_2d(data_augmentation_kwargs=None, label_warp=None,
                           label_shift=None, label_dropout=0,
                           label_crop_rand=None, label_crop_rand2=None,
                           label_crop_left=None):
    """
    Preprocessor function to pass to a data_flow, for BRATS data.
    
    data_augmentation_kwargs : Dictionary of keyword arguments to pass to
        the data augmentation code (image_stack_random_transform).
    label_warp (float) : The sigma value of the spline warp applied to
        to the target label mask during training in order to corrupt it. Used
        for testing robustness to label noise.
    label_shift (int) : The number of pixels to shift all training target masks
        to the right.
    label_dropout (float) : The probability in [0, 1] of discarding a slice's
        segmentation mask.
    label_crop_rand (float) : Crop out a randomly sized rectangle out of every
        connected component of the mask. The minimum size of the rectangle is
        set as a fraction of the connected component's bounding box, in [0, 1].
    label_crop_rand2 (float) : Crop out a randomly sized rectangle out of every
        connected component of the mask. The mean size in each dimension is
        set as a fraction of the connected component's width/height, in [0, 1].
    label_crop_left (float) : If true, crop out the left fraction (in [0, 1]) 
        of every connected component of the mask.
    """
        
    def process_element(inputs):

        source, source_seg, target = inputs
        

        # Float.
        source = source.astype(np.float32)
        target = target.astype(np.float32)

                    
        # Data augmentation 2D.
        if data_augmentation_kwargs is not None:
            if target is not None:
                target = image_random_transform(target, **data_augmentation_kwargs,
                                               n_warp_threads=1)        
                
            if source is not None:
                _ = image_random_transform(source, source_seg, **data_augmentation_kwargs,
                                               n_warp_threads=1)
            if source_seg is not None:
                assert source is not None
                source, source_seg = _
            else:
                source = _    
        
        # Remove distant outlier intensities.
        if source is not None:
            source = np.clip(source, -1., 1.) 
        if target is not None:
            target = np.clip(target, -1., 1.) 
   
        return source, source_seg, target

    def process_batch(batch):
        # Process every element.
        elements = []
        for i in range(len(batch[0])):
            elem = process_element([b[i] for b in batch])
            elements.append(elem)
        out_batch = list(zip(*elements))
        return out_batch
    
    return process_batch
