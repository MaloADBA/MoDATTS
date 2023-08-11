
from __future__ import (print_function,
                        division)
import os
import argparse
from collections import OrderedDict
#from concurrent.futures import ThreadPoolExecutor
import threading
try:
    import queue            # python 3
except ImportError:
    import Queue as queue   # python 2

import numpy as np
import scipy.misc
from scipy import ndimage
import SimpleITK as sitk
import h5py
import imageio
import random


def parse():
    parser = argparse.ArgumentParser(description="Prepare BRATS data. Loads "
        "BRATS 2020 data and stores volume slices in an HDF5 archive. "
        "Slices are organized as a group per patient, containing three "
        "groups: \'sick\', \'healthy\', and \'segmentations\' for each of the 2 modalities."
        "Sick cases contain any anomolous class, healthy cases contain no anomalies, and "
        "segmentations are the segmentations of the anomalies. Each group "
        "contains subgroups for each of the three orthogonal planes along "
        "which slices are extracted. For each case, MRI sequences are stored "
        "in a single volume")
    parser.add_argument('--data_dir',
                        help="The directory containing the BRATS 2020 data. ",
                        required=True, type=str)
    parser.add_argument('--save_to',
                        help="Path to save the HDF5 file to.",
                        required=True, type=str)
    parser.add_argument('--num_threads',
                        help="The number of parallel threads to execute.",
                        required=False, type=int, default=None)
    parser.add_argument('--save_debug_to',
                        help="Save images of each slice to this directory, "
                             "for inspection.",
                        required=False, type=str, default=None)
    return parser.parse_args()


def data_loader(data_dir):
    tags = ['flair', 't1ce', 't1', 't2', 'seg']
    tags_modality = ['flair', 't1ce', 't1', 't2', 'seg']
    for dn in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, dn)
        
        # Match filenames to tags.
        # NOTE: t1ce chosen when both t1 and t1ce matched
        fn_dict = OrderedDict()
        for fn in sorted(os.listdir(path)):
            match = [t for t in tags if t in fn]
            fn_dict[match[0]] = fn
        print(fn_dict)    
        # Load files.
        vol_all = []
        segmentation = None
        size = None
        size = None
        for t in tags_modality:
            vol = sitk.ReadImage(os.path.join(path, fn_dict[t]))
            vol_np = sitk.GetArrayFromImage(vol)
            if size is None:
                size = vol_np.shape
            if vol_np.shape != size:
                raise Exception("Expected {} to have a size of {} but got {}."
                                "".format(fn_dict[t], size, vol_np.shape))
            
            if t=='seg':
                segmentation = vol_np.astype(np.int64)
                segmentation = np.expand_dims(segmentation, 0)
            else:
                vol_np = vol_np.astype(np.float32)
                vol_all.append(np.expand_dims(vol_np, 0))
                
        # Concatenate on channel axis.
        volume = np.concatenate(vol_all, axis=0)
            
        yield volume, segmentation, dn
        

        
        
def get_slices(volume, segmentation):
    
    # Axis transpose order.
    axis = 1
    order = [1,0,2,3]
    volume = volume.transpose(order)
    segmentation = segmentation.transpose(order)

    # Sort slices.
    slices_dict = OrderedDict()
    slices_dict['flair'] = np.pad(np.expand_dims(volume[30:126,0,:,:],axis=1), [(0, 0), (0, 0), (8, 8), (8, 8)], mode='constant', constant_values=0)
    slices_dict['t1ce'] = np.pad(np.expand_dims(volume[30:126,1,:,:],axis=1), [(0, 0), (0, 0), (8, 8), (8, 8)], mode='constant', constant_values=0)
    slices_dict['t1'] = np.pad(np.expand_dims(volume[30:126,2,:,:],axis=1), [(0, 0), (0, 0), (8, 8), (8, 8)], mode='constant', constant_values=0)
    slices_dict['t2'] = np.pad(np.expand_dims(volume[30:126,3,:,:],axis=1), [(0, 0), (0, 0), (8, 8), (8, 8)], mode='constant', constant_values=0)
    slices_dict['segmentation'] = np.pad(segmentation[30:126], [(0, 0), (0, 0), (8, 8), (8, 8)], mode='constant', constant_values=0)
    print(slices_dict['t2'].shape)
    print(slices_dict['segmentation'].shape)
    return slices_dict


def preprocess(volume, segmentation):
    volume_out = volume.copy()
    segmentation_out = segmentation.copy()
    
    # Mean center and normalize by std.
    brain_mask = volume!=0
    volume_out[brain_mask] -= volume_out[brain_mask].mean()
    volume_out[brain_mask] /= volume_out[brain_mask].std()*5    # fit in tanh
       
    return volume_out, segmentation_out


def process_case(case_num, h5py_file, volume, segmentation, fn,
                 save_debug_to=None):
    print("Processing case {}: {}".format(case_num, fn))
    group_p = h5py_file.create_group(str(case_num))
    # TODO: set attribute containing fn.
    print('preprocessing')
    volume, seg = preprocess(volume, segmentation)
    
    print('getting_slices')
    slices = get_slices(volume, seg)

    for key in slices.keys():
        if len(slices[key])==0:
            kwargs = {}
        else:
            kwargs = {'chunks': (1,)+slices[key].shape[1:],
                      'compression': 'lzf'}
        group_p.create_dataset(key,
                               shape=slices[key].shape,
                               data=slices[key],
                               dtype=slices[key].dtype,
                               **kwargs)
    
    # Debug outputs for inspection.
    if save_debug_to is not None:
        for key in slices.keys():
            if "indices" in key:
                continue
            dest = os.path.join(save_debug_to, key)
            if not os.path.exists(dest):
                os.makedirs(dest)
            for i in range(len(slices[key])):
                im = slices[key][i]
                for ch, im_ch in enumerate(im):
                    imageio.imwrite(os.path.join(dest, "{}_{}_{}.png"
                                                   "".format(case_num, i, ch)),
                                      slices[key][i][ch])
                                       

if __name__=='__main__':
    args = parse()
    #if os.path.exists(args.save_to):
        #raise ValueError("Path to save data already exists. Aborting.")
    h5py_file = h5py.File(args.save_to, mode='w')
    for i, (volume, seg, fn) in enumerate(data_loader(args.data_dir)):
        process_case(i, h5py_file, volume, seg, fn,
                        args.save_debug_to)

