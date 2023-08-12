
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

from mbrats_3d_self_training import get_parser as get_model_parser_self_training
import model.gen_segmentation_ulti_nmsc_dif
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
                        help="Source modality", type=int, default='1')  
    parser.add_argument('--threshold',
                        help="Threshold for label sharpening", type=float, default=0.6)      
    parser.add_argument('--num_threads',
                        help="The number of parallel threads to execute.",
                        required=False, type=int, default=None)
    return parser.parse_args()

def load_model(experiment_path, best=True, mtype = 'M_Gen', model_kwargs=None):
    # Load args.
    print("Loading experiment arguments.")
    args_path = os.path.join(experiment_path, 'args.txt')
    model_parser = get_model_parser_self_training()
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

if __name__=='__main__':
    args = parse()
    #if os.path.exists(args.save_to):
        #raise ValueError("Path to save data already exists. Aborting.")
    h5py_file = h5py.File(args.save_to, mode='w')
    data = h5py.File(args.data_dir, mode='r')
    
    model = load_model(experiment_path=args.model_path)
    model.eval()    
    
    if args.source == 0:
        real = 'target'
        fake = 'source_to_target'
    elif args.source == 1:
        real = 'source'
        fake = 'target_to_source'  
        
    for case_id in data.keys():
        dic = OrderedDict()
        print("Processing case "+case_id)
        group_p = h5py_file.create_group(case_id)
        if not len(data[case_id]['sick_real_'+real]):
            dic['pseudo_labels'] = np.zeros((1,1,240,120))[[],:]
        else:
            pseudo_labels = []
            for i in range(len(data[case_id]['sick_real_'+real])):
                x_TA = torch.unsqueeze(torch.from_numpy(np.array(np.clip(data[case_id]['sick_real_'+real][i].astype(np.float32), -1., 1.))).cuda(),dim=0)
                mask_T = [np.expand_dims(data[case_id]['sick_real_'+real][i,0],0)]
            
                with torch.no_grad():
                    outputs = model(x_SAT=x_TA, x_TA=x_TA, x_TB=x_TA, mask_S=mask_T, mask_T=mask_T)            
                    pseudo_labels.append((np.squeeze(outputs['x_TAM_prob'].cpu().detach().numpy())>args.threshold)*1)           
            dic['pseudo_labels'] = np.expand_dims(np.array(pseudo_labels),axis=1)
            print(dic['pseudo_labels'].shape)
        for key in dic.keys() :
            if len(dic[key])==0:
                kwargs = {}
            else:
                kwargs = {'chunks': (1,)+dic[key].shape[1:],
                          'compression': 'lzf'}
            group_p.create_dataset(key,
                                   shape=dic[key].shape,
                                   data=dic[key],
                                   dtype=dic[key].dtype,
                               **kwargs)

