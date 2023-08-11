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

def prepare_mbrats_trans_3d(path,
                   pseudo_labels,         
                   masked_fraction_1=0, masked_fraction_2=0, 
                   drop_masked=False, rng=None):
    # Random 20% data split (10% validation, 10% testing).
    rnd_state = np.random.RandomState(0)
    indices = np.arange(0, 369)
    rnd_state.shuffle(indices)
    val  = indices[0:37]
    test = indices[37:74]
    return _prepare_mbrats_trans_3d(path,
                              pseudo_labels,
                              masked_fraction_1=masked_fraction_1, 
                              masked_fraction_2=masked_fraction_2,                               
                              validation_indices=val,
                              testing_indices=test,
                              drop_masked=drop_masked,
                              rng=rng)

def _prepare_mbrats_trans_3d(path, pseudo_labels, validation_indices, testing_indices,
                             masked_fraction_1=0, masked_fraction_2=0, drop_masked=False, rng=None):
    
    if rng is None:
        rng = np.random.RandomState(0)
    
    # Assemble volumes and corresponding segmentations; split train/valid/test.
    volumes_real_h = {'train': [], 'valid': [], 'test': []}
    volumes_real_s = {'train': [], 'valid': [], 'test': []}
    volumes_fake_s = {'train': [], 'valid': [], 'test': []}
    volumes_real_m = {'train': [], 'valid': [], 'test': []}
    volumes_fake_m = {'train': [], 'valid': [], 'test': []}
    pseudo = []

    h5py_file_data = h5py.File(path, mode='r')
    if pseudo_labels is not None:
        h5py_file_pl = h5py.File(pseudo_labels, mode='r')
    
    indices = []
    case_identities = []
    for idx, case_id in enumerate(h5py_file_data.keys()):   # Per patient.
        indices.append(idx)
        case_identities.append(case_id)
    c = list(zip(indices,  case_identities))
    
    for idx, case_id in c:
        
        f = h5py_file_data[case_id]
        if pseudo_labels is not None:
            pl = h5py_file_pl[case_id]
        if idx in validation_indices:
            split = 'valid'
            volumes_real_h[split].append(f['healthy_real_target'])
            volumes_real_s[split].append(f['sick_real_target'])
            volumes_real_m[split].append(f['segmentation'])
        elif idx in testing_indices:
            split = 'test'
            volumes_real_h[split].append(f['healthy_real_target'])
            volumes_real_s[split].append(f['sick_real_target'])
            volumes_real_m[split].append(f['segmentation'])         
        else:
            split = 'train'
            volumes_real_h[split].append(f['healthy_real_target'])
            volumes_real_s[split].append(f['sick_real_target'])
            volumes_real_m[split].append(f['segmentation'])
            if pseudo_labels is not None:
                pseudo.append(pl['pseudo_labels'])
    
    for idx, case_id in c:

        f = h5py_file_data[case_id]
        if idx in validation_indices:
            split = 'valid'
            volumes_fake_s[split].append(f['sick_source_to_target'])
            volumes_fake_m[split].append(f['segmentation'])
        elif idx in testing_indices:
            split = 'test'
            volumes_fake_s[split].append(f['sick_source_to_target'])
            volumes_fake_m[split].append(f['segmentation'])           
        else:
            split = 'train'
            volumes_fake_s[split].append(f['sick_source_to_target'])
            volumes_fake_m[split].append(f['segmentation'])
            
    if masked_fraction_1!=1:
        # source modality
        # Volumes with these indices will either be dropped from the training
        # set or have their segmentations set to None.
        # 
        # The `masked_fraction` determines the maximal fraction of slices that
        # are to be thus removed. All or none of the slices are selected for 
        # each volume.
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_fake_m['train']])
        num_masked_slices = 0
        max_masked_slices = int(min(num_total_slices,
                                    num_total_slices*masked_fraction_1+0.5))
        for i in rng.permutation(len(volumes_fake_m['train'])):
            num_slices = len(volumes_fake_m['train'][i])
            if num_slices>0 and num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            if num_slices+num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            masked_indices.append(i)
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices)+"source slices are labeled across {} "
              "volumes ({:.1f}%).".format(len(volumes_fake_m['train'])-len(masked_indices),
                        100*(1-num_masked_slices/float(num_total_slices))))
    else : 
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_fake_m['train']])
        num_masked_slices = 0
        for i in rng.permutation(len(volumes_fake_m['train'])):
            num_slices = len(volumes_fake_m['train'][i])
            masked_indices.append(i)
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices,
                        )+"source slices are labeled"
              )
    
    # Apply masking in one of two ways.
    # 
    # 1. Mask out the labels for volumes indexed with `masked_indices` by 
    # setting the segmentations volume as an array of `None`, with length 
    # equal to the number of slices in the volume.
    # 
    # OR if `drop_masked` is True:
    # 
    # 2. Remove all volumes indexed with `masked_indices`.
    volumes_fake_s_train = []
    volumes_fake_m_train = []
    for i in range(len(volumes_fake_m['train'])):
        if i in masked_indices:
            # Mask out or drop.
            if drop_masked:
                continue    # Drop.
            volumes_fake_m_train.append(np.array([None]*len(volumes_fake_m['train'][i])))
        else:
            # Keep.
            volumes_fake_m_train.append(volumes_fake_m['train'][i])
        volumes_fake_s_train.append(volumes_fake_s['train'][i])
    volumes_fake_s['train'] = volumes_fake_s_train
    volumes_fake_m['train'] = volumes_fake_m_train   

    if masked_fraction_2!=1:
        # target modality
        # Volumes with these indices will either be dropped from the training
        # set or have their segmentations set to None.
        # 
        # The `masked_fraction` determines the maximal fraction of slices that
        # are to be thus removed. All or none of the slices are selected for 
        # each volume.
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_real_m['train']])
        num_masked_slices = 0
        max_masked_slices = int(min(num_total_slices,
                                    num_total_slices*masked_fraction_2+0.5))
        for i in rng.permutation(len(volumes_real_m['train'])):
            num_slices = len(volumes_real_m['train'][i])
            if num_slices>0 and num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            if num_slices+num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            masked_indices.append(i)
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices)+"target slices are labeled across {} "
              "volumes ({:.1f}%).".format(len(volumes_real_m['train'])-len(masked_indices),
                        100*(1-num_masked_slices/float(num_total_slices))))
    else : 
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_real_m['train']])
        num_masked_slices = 0
        for i in rng.permutation(len(volumes_real_m['train'])):
            masked_indices.append(i)
            num_slices = len(volumes_real_m['train'][i])
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices,
                        )+"target slices are labeled"
              )
    
    # Apply masking in one of two ways.
    # 
    # 1. Mask out the labels for volumes indexed with `masked_indices` by 
    # setting the segmentations volume as an array of `None`, with length 
    # equal to the number of slices in the volume.
    # 
    # OR if `drop_masked` is True:
    # 
    # 2. Remove all volumes indexed with `masked_indices`.
    volumes_real_s_train = []
    volumes_real_h_train = []
    volumes_real_m_train = []
    for i in range(len(volumes_real_m['train'])):
        if i in masked_indices:
            # Mask out or drop.
            if drop_masked:
                continue    # Drop.
            elif pseudo_labels is not None:
                volumes_real_m_train.append(pseudo[i])
            else:
                volumes_real_m_train.append(np.array([None]*len(volumes_real_m['train'][i])))
        else:
            # Keep.
            volumes_real_m_train.append(volumes_real_m['train'][i])
        volumes_real_h_train.append(volumes_real_h['train'][i])
        volumes_real_s_train.append(volumes_real_s['train'][i])
    volumes_real_h['train'] = volumes_real_h_train
    volumes_real_s['train'] = volumes_real_s_train
    volumes_real_m['train'] = volumes_real_m_train

    # Merge all arrays in each list of arrays.
    data = OrderedDict([('train', OrderedDict()),
                        ('valid', OrderedDict()),
                        ('test', OrderedDict())])
    for key in data.keys():
        # HACK: we may have a situation where the number of sick examples
        # is greater than the number of healthy. In that case, we should
        # duplicate the healthy set M times so that it has a bigger size
        # than the sick set.
        m = 1
        len_h = sum([len(elem) for elem in volumes_real_h[key]])
        len_s = sum([len(elem) for elem in volumes_real_s[key]])
        if len_h < len_s:
            m = int(np.ceil(len_s / len_h))
        data[key]['real_h']  = multi_source_array(volumes_real_h[key]*m)
        data[key]['real_s']  = multi_source_array(volumes_real_s[key])
        data[key]['real_m']  = multi_source_array(volumes_real_m[key])
        data[key]['fake_s']  = multi_source_array(volumes_fake_s[key])
        data[key]['fake_m']  = multi_source_array(volumes_fake_m[key])
    return data



def preprocessor_mbrats_trans_3d(data_augmentation_kwargs=None, tumor_augmentation=None, label_warp=None,
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

        real_h, real_s, real_m, fake_s, fake_m = inputs
       
        # Float.
        real_h = real_h.astype(np.float32)
        real_s = real_s.astype(np.float32)
        fake_s = fake_s.astype(np.float32)
        
        
        # Tumor intensity augmentation      
        if tumor_augmentation and fake_m is not None:
            lambda_aug = np.random.uniform(low=0.7,high=1.3)
            fake_s[fake_m!=0]*=lambda_aug
        
        # Data augmentation 3D.
        if data_augmentation_kwargs is not None:
            if real_h is not None:
                real_h = nnunet_transform_default_3d(real_h)         
                
            if real_s is not None:
                _ = nnunet_transform_default_3d(real_s, real_m)
            if real_m is not None:
                assert real_s is not None
                real_s, real_m = _
            else:
                real_s = _
            if fake_s is not None:
                _ = nnunet_transform_default_3d(fake_s, fake_m)
            if fake_m is not None:
                assert fake_s is not None
                fake_s, fake_m = _
            else:
                fake_s = _                
        
        # Remove distant outlier intensities.
        if real_h is not None:
            real_h = np.clip(real_h, -1., 1.)
        if real_s is not None:
            real_s = np.clip(real_s, -1., 1.)

        if fake_s is not None:
            fake_s = np.clip(fake_s, -1., 1.)  
            
        return real_h, real_s, real_m, fake_s, fake_m        

    def process_batch(batch):
        # Process every element.
        elements = []
        for i in range(len(batch[0])):
            elem = process_element([b[i] for b in batch])
            elements.append(elem)
        out_batch = list(zip(*elements))
        return out_batch
    
    return process_batch



def nnunet_transform_default_3d(img, seg=None, border_val_seg=-1, order_seg=0, order_data=3):
    params = {'selected_data_channels': None, 'selected_seg_channels': [0], 'do_elastic': False,
              'elastic_deform_alpha': (0.0, 900.0), 'elastic_deform_sigma': (9.0, 13.0), 'do_scaling': True,
              'scale_range': (0.7, 1.4), 'do_rotation': True, 'rotation_x': (-0.5235987755982988, 0.5235987755982988),
              'rotation_y': (-0.5235987755982988, 0.5235987755982988),
              'rotation_z': (-0.5235987755982988, 0.5235987755982988), 'random_crop': False,
              'random_crop_dist_to_border': None, 'do_gamma': True, 'gamma_retain_stats': True,
              'gamma_range': (0.7, 1.5), 'p_gamma': 0.3, 'num_threads': 12, 'num_cached_per_thread': 1,
              'do_mirror': True, 'mirror_axes': (0, 1, 2), 'p_eldef': 0.2, 'p_scale': 0.2, 'p_rot': 0.2,
              'dummy_2D': False,
              'mask_was_used_for_normalization': OrderedDict([(0, True), (1, True), (2, True), (3, True)]),
              'all_segmentation_labels': None, 'move_last_seg_chanel_to_data': False, 'border_mode_data': 'constant',
              'cascade_do_cascade_augmentations': False,
              'patch_size_for_spatialtransform': np.array([128, 128, 128])}
    ignore_axes = None

    transforms = []

    transforms += [
        SpatialTransform(
            None, patch_center_dist_from_border=None,
            do_elastic_deform=params.get("do_elastic"), alpha=params.get("elastic_deform_alpha"),
            sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"),
            do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg,
            order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=True, p_rot_per_axis=1)
    ]

    transforms += [
        GaussianNoiseTransform(p_per_sample=0.1)
    ]

    transforms += [
        GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                              p_per_channel=0.5)
    ]

    transforms += [
        BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15)
    ]

    if params.get("do_additive_brightness"):
        transforms += [
            BrightnessTransform(params.get("additive_brightness_mu"),
                                params.get("additive_brightness_sigma"),
                                True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                p_per_channel=params.get("additive_brightness_p_per_channel"))
        ]

    transforms += [ContrastAugmentationTransform(p_per_sample=0.15)]
    transforms += [
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                       p_per_channel=0.5,
                                       order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                       ignore_axes=ignore_axes)
    ]

    transforms += [
        GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                       p_per_sample=0.1)
    ]

    if params.get("do_gamma"):
       transforms += [
            GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"])
       ]

    if params.get("do_mirror") or params.get("mirror"):
        transforms += [
            MirrorTransform((0,1))
        ]

    #if params.get("mask_was_used_for_normalization") is not None:
    #    mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
    #    transforms += [
    #        MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0)
    #    ]

    full_transform = Compose(transforms)

    # Transform.
    img_input = img[None]
    seg_input = seg
    if seg is not None:
        seg_input = seg[None]
    out = full_transform(data=img_input, seg=seg_input)
    img_output = out['data'][0]
    if seg is None:
        return img_output
    seg_output = out['seg'][0]
    return img_output, seg_output


