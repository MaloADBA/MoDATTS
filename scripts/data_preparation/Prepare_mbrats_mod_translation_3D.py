
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
import torch
import numpy as np
import scipy.misc
from scipy import ndimage
import SimpleITK as sitk
import h5py
import imageio
import random

from mbrats_2d_trans import get_parser as get_model_parser
from utils.experiment import experiment
from torch.autograd import Variable

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
    parser.add_argument('--model_path',
                        help="Path to translation model 2D.",
                        required=True, type=str)    
    parser.add_argument('--source',
                        help="Source modality",
                        required=True, type=str, default='t2')   
    parser.add_argument('--target',
                        help="Target modality",
                        required=True, type=str, default='t1')                          
    parser.add_argument('--num_threads',
                        help="The number of parallel threads to execute.",
                        required=False, type=int, default=None)
    return parser.parse_args()

def load_model(experiment_path, best=True,  model_kwargs=None):
    # Load args.
    print("Loading experiment arguments.")
    args_path = os.path.join(experiment_path, 'args.txt')
    model_parser = get_model_parser()
    with open(args_path, 'r') as f:
        saved_args = f.read().split('\n')[1:]
        saved_args[saved_args.index('--path')+1] = experiment_path
        if model_kwargs is not None:
            saved_args[saved_args.index('--model_kwargs')+1] = model_kwargs
        model_args = model_parser.parse_args(args=saved_args)
    experiment_state = experiment(model_args)
    if best:
        print("Loading checkpoint at best epoch by validation score.")
        experiment_state.load_best_state()
    model = experiment_state.model['G']
    return model

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
 
""""Crop or pad to the right dimensions functions"""

def pad_or_crop_image(image, target_size=(128, 128, 128), random_crop = False):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim, random_crop = random_crop) for target, dim in zip(target_size, (z, y, x))]
    image = image[:, z_slice, y_slice, x_slice]
    todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    padlist = [(0, 0)]  # channel dim
    for to_pad in todos:
        if to_pad[0]:
            padlist.append((to_pad[1], to_pad[2]))
        else:
            padlist.append((0, 0))
    image = np.pad(image, padlist)
    return image
    
def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    elif dim < target_size:
        pad_extent = target_size - dim
        left = int(pad_extent/2)
        right = pad_extent - left
        return True, left, right


def get_crop_slice(target_size, dim, random_crop):
    if dim > target_size:
        crop_extent = dim - target_size
        if random_crop:
            left = random.randint(0, crop_extent)
        else:
            left = int(crop_extent/2) #
        right = crop_extent - left
        return slice(left, dim - right)
    elif dim <= target_size:
        return slice(0, dim)    
        
def translate(volume, segmentation, model, source, target):
    volume_out = volume.copy()
    segmentation_out = segmentation.copy()
    
    bbox = ndimage.find_objects(volume!=0)[0]
    
    # Mean center and normalize by std.
    brain_mask = volume!=0
    volume_out[brain_mask] -= volume_out[brain_mask].mean()
    volume_out[brain_mask] /= volume_out[brain_mask].std()*5    # fit in tanh

    source_to_target = np.zeros_like(volume_out)
    
    modalities = ['flair', 't1ce', 't1', 't2']
    for j in range(4):
        if source == modalities[j]:
                i_source = j
        if target == modalities[j]:
                i_target = j	

    dic = OrderedDict()
    dic['source_to_target'] = np.zeros(volume_out[i_source][30:126,:,:].shape)
    dic['real_target'] = volume_out[i_target][30:126,:,:]
    dic['segmentation'] = segmentation_out              
                
    for i in range(6):
        batch = torch.unsqueeze(torch.from_numpy(np.pad(np.clip(volume_out[i_source,30+i*16:30+(i+1)*16,:,:].astype(np.float32), -1., 1.), [(0, 0), (8, 8), (8, 8)], mode='constant', constant_values=0)),dim=1).cuda()
        with torch.no_grad():
                outputs = model(x_SA=batch, x_TA=batch, mask_S=[None]*16) 
                
        dic['source_to_target'][i*16:(i+1)*16,:,:] = np.squeeze(outputs['x_SAT'][:,0,8:248,8:248].cpu().detach().numpy())
      

    #crop to brain area
    dic['source_to_target'] = dic['source_to_target'][:,bbox[2],:]
    dic['real_target'] = dic['real_target'][:,bbox[2],:]
    dic['segmentation'] = np.squeeze(dic['segmentation'])[:,bbox[2],:]
    
    return dic
        
def process_case(case_num, h5py_file, volume, segmentation, fn, model, source, target):

    group_p = h5py_file.create_group(str(case_num))
    
    print('processing')
    dic = translate(volume, segmentation, model, source, target)
    
    # Split volume along hemispheres.
    mid0 = dic['source_to_target'].shape[-1]//2
    mid1 = mid0
    if volume.shape[-1]%2:
        mid0 += 1     
        
    seg1 = dic['segmentation'][:,:,:mid0] 
    seg2 = dic['segmentation'][:,:,mid1:]  

    
    # Identify healthy, sick.
    hemispheres = {'healthy_real_target': [], 'sick_real_target': [], 'sick_source_to_target': []}
                   
    if np.any(seg1>0):
        
        hemispheres['sick_real_target'].append(pad_or_crop_image(np.expand_dims(dic['real_target'][:,:,mid0-64:mid0] ,axis=0), target_size=(96, 160,64), random_crop = False))
        hemispheres['sick_source_to_target'].append(pad_or_crop_image(np.expand_dims(dic['source_to_target'][:,:,mid0-64:mid0] ,axis=0), target_size=(96, 160,64), random_crop = False))
        hemispheres['segmentation'].append(pad_or_crop_image(np.expand_dims(dic['segmentation'][:,:,mid0-64:mid0] ,axis=0), target_size=(96, 160,64), random_crop = False))

    else:

        hemispheres['healthy_real_target'].append(pad_or_crop_image(np.expand_dims(dic['real_target'][:,:,mid0-64:mid0] ,axis=0), target_size=(96, 160,64), random_crop = False))
            
    if np.any(seg2>0):

        hemispheres['sick_real_target'].append(pad_or_crop_image(np.expand_dims(dic['real_target'][:,:,mid1:mid1+64] ,axis=0), target_size=(96, 160,64), random_crop = False))
        hemispheres['sick_source_to_target'].append(pad_or_crop_image(np.expand_dims(dic['source_to_target'][:,:,mid1:mid1+64] ,axis=0), target_size=(96, 160,64), random_crop = False))
        hemispheres['segmentation'].append(pad_or_crop_image(np.expand_dims(dic['segmentation'][:,:,mid1:mid1+64] ,axis=0), target_size=(96, 160,64), random_crop = False))

    else:

        hemispheres['healthy_real_target'].append(pad_or_crop_image(np.expand_dims(dic['real_target'][:,:,mid1:mid1+64] ,axis=0), target_size=(96, 160,64), random_crop = False))

    stacks = {}
    
    for key in hemispheres:
        if len(hemispheres[key]) == 0:
            stacks[key] = np.zeros((0,0,0,0,0), dtype=volume.dtype)
        else:
            stacks[key] = np.stack(hemispheres[key])  
            print(stacks[key].shape)  
    
    #save
    kwargs = {'compression': 'lzf'}
    
    for key in stacks.keys():
        group_p.create_dataset(key,
                               shape=stacks[key].shape,
                               data=stacks[key],
                               dtype=stacks[key].dtype,
                               **kwargs)      
    
                                       

if __name__=='__main__':
    args = parse()
    #if os.path.exists(args.save_to):
        #raise ValueError("Path to save data already exists. Aborting.")
    h5py_file = h5py.File(args.save_to, mode='w')
    model = load_model(experiment_path=args.model_path).cuda()
    model.eval()
    for i, (volume, seg, fn) in enumerate(data_loader(args.data_dir)):
        process_case(i, h5py_file, volume, seg, fn, model, args.source, args.target)

