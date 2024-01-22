################
#written by Federico Spagnolo
#usage: python ies.py --model_checkpoint model_epoch_31.pth --input_val_paths batch_data batch_data --input_prefixes flair_3d_sbr.nii.gz t1n_3d_sb.nii.gz --num_workers 0 --cache_rate 0.01 --threshold 0.3
################
# Import torch
import torch
import torchvision
#import packages for model description
from torchvision import models
import scipy.ndimage as ndimage
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import UNet
from monai.transforms import (Activations, AddChanneld, Compose, ConcatItemsd,
                              LoadImaged, NormalizeIntensityd, RandAffined,
                              RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
                              RandShiftIntensityd, RandSpatialCropd, Spacingd, SelectItemsd,
                              ToTensord)
from monai.networks.nets import UNet
from monai.visualize import GradCAMpp
from datasets import NiftinotargetDataset
from transforms import remove_connected_components, get_valnotarget_transforms, binarize_mask
from losses import *
from lesion_extraction import get_lesion_types_masks
import numpy as np
import nibabel as nib
import pathlib
import argparse
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy.ma as ma
from tqdm import tqdm
import pandas as pd
import gc
import time
import psutil

def dp(path1: str, path2: str) -> str:
    return os.path.join(path1, path2)

def form_cluster(data_array:np.array, struct:np.array=np.ones([3, 3, 3])):
    """Get individual clusters

    Args:
        data_array (numpy array): The image, where to find clusters
        struct (numpy array or scipy struct array, optional): The connectivity. Defaults to np.ones([3, 3, 3]) for all-direction connectivity.

    Returns:
        label_map [numpy array]: The image having labeled clusters.
        unique_label [numpy array]: The array containing unique cluster indices.
        label_counts [numpy array]: The correpsonding voxel numbers.
    """
    label_map, _ = ndimage.label(data_array, structure=struct)
    unique_label, label_counts = np.unique(label_map, return_counts=True)
    return label_map, unique_label[1:], label_counts[1:]    
    
# current folder
script_folder = os.path.dirname(os.path.realpath(__file__))
# set output folder
output_dir = "attention_maps"

parser = argparse.ArgumentParser(
description='''Saliency map generation.
                        If no arguments are given, all the default values will be used.''', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='-d', dest='data_path', default=script_folder, help="Path where the data is")
parser.add_argument('--model_checkpoint', metavar='-ckpt', dest='model_checkpoint', default="model_epoch_1.pth", help="Path to he best checkpoint of the model")
parser.add_argument('--input_val_paths', type=str, nargs='+', required=True)
parser.add_argument('--input_prefixes', type=str, nargs='+', required=True)
# data handling
parser.add_argument('--num_workers', type=int, default=0,
                        help='number of jobs used to load the data, DataLoader parameter')
parser.add_argument('--cache_rate', type=float, default=0.1, help='cache dataset parameter')
parser.add_argument('--threshold', type=float, default=0.5, help='lesion threshold')
args = parser.parse_args()        

data_path = args.data_path
model_checkpoint = args.model_checkpoint
args.input_modalities = ['flair', 'mprage']
args.n_classes = 2
seed = 1

''' Get default device '''
logging.basicConfig(level = logging.INFO)
print("total devices", torch.cuda.device_count())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logging.info(f"Using device: {device}")
torch.multiprocessing.set_sharing_strategy('file_system')

# Init your model
model = UNet(spatial_dims=3, in_channels=len(args.input_modalities), out_channels=args.n_classes,
                     channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), norm='batch', num_res_units=0).cuda()
# Weights intialization
for layer in model.model.modules():
        if type(layer) == torch.nn.Conv3d:
                torch.nn.init.xavier_normal_(layer.weight, gain=1.0)

# Load best model weights
model_path = dp(script_folder, model_checkpoint)
#print(model_path)
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load(args.model_checkpoint, map_location='cuda'))
model.eval()
activation = torch.nn.Softmax(dim=1)

''' Define validation actors '''
inferer = SlidingWindowInferer(roi_size=(96, 96, 96),
                                   sw_batch_size=1, mode='gaussian', overlap=0.25)

logging.info(f"Best model save file loaded from: {model_path}")

# Load Dataset
val_transforms = get_valnotarget_transforms(input_keys=args.input_modalities).set_random_state(seed=seed)
val_dataset = NiftinotargetDataset(input_paths=args.input_val_paths, input_prefixes=args.input_prefixes,
                                   input_names=args.input_modalities, 
                                   transforms=val_transforms, num_workers=args.num_workers,
                                   cache_rate=args.cache_rate)                                   
                                   
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=args.num_workers)

loss_function = NormalisedDiceFocalLoss(include_background=False, to_onehot_y=True, other_act=activation,
                                                lambda_ndscl=0.5, lambda_focal=1.0)

i = 0
threshold = args.threshold
std_fraction = 0.05
nb_smooth = 50

layer = "module.model.1.submodule.2.conv"
for name, _ in model.named_modules(): print(name)
cam = GradCAMpp(nn_module=model, target_layers=layer)
logging.info(f"Initializing the dataset. Number of subjects {len(val_dataloader)}")

# Batch Loading

for data in val_dataloader:

     filename = data['flair_meta_dict']['filename_or_obj'][0]
     logging.info(f"Evaluate gradients in batch {filename}")
     inputs = data["inputs"].cuda() # 0 is flair, 1 is mprage
     std = std_fraction * (inputs[0,0].max() - inputs[0,0].min())
     input_affine = nib.load(filename).affine

     inputs.requires_grad_()
     
     #outputs = model(inputs)
     outputs = inferer(inputs=inputs, network=model)  # [1, 2, H, W, D]
     outputs = activation(outputs)  # [1, 2, H, W, D]
     output_mask = outputs[0,1].detach().cpu().numpy()
     output_mask[output_mask > threshold] = 1
     output_mask[output_mask < threshold] = 0

     # Save predicted output
     pred = nib.Nifti1Image(output_mask, input_affine)
     outputname = (filename.split("data/", 1)[1]).split("/flair", 1)[0].replace('/', '-') + ".nii.gz"
     patientname = (filename.split("data/", 1)[1]).split("/flair", 1)[0].replace('/', '-')
     
     try:
            os.mkdir('./' + output_dir + '/' + patientname)
     except FileExistsError:
            logging.info(f"Directory {'./' + output_dir + '/' + patientname} already exists")

     nib.save(pred, "{}/{}/pred_{}".format(output_dir, patientname, outputname))
     
     wm_labels = get_lesion_types_masks(output_mask, output_mask, 'non_zero', n_jobs = 1)['TPL']        
     selected_group = nib.Nifti1Image(wm_labels, input_affine)

     logging.info(f"Detected lesions: {np.max(wm_labels)}")
     label = 1
     while label <= np.max(wm_labels):
          patch_vector = np.where(wm_labels==label)
          if len(patch_vector[0])<=5:
               wm_labels[wm_labels==label] = 0
               wm_labels[wm_labels>label] = wm_labels[wm_labels>label] - 1
          else:
               label = label + 1     

     n_labels = np.max(wm_labels)
     logging.info(f"Detected lesions > 5mm³: {n_labels}")
     
     selected_group = nib.Nifti1Image(wm_labels, input_affine)
     nib.save(selected_group, "{}/{}/group_{}".format(output_dir, patientname, outputname))
     
     # Generate and save GCAM++
     print("Generating XAI maps for method 1: GradCAM++...")
     for label in range(1, n_labels):
          patch_vector = np.where(wm_labels==label)
          gcam = cam(x=inputs, class_idx=1, footprint=patch_vector)
          gcam_map = nib.Nifti1Image(gcam[0,0].cpu().numpy(), input_affine)
          nib.save(gcam_map, "{}/{}/GradCAM_lesion{}_layer_{}_{}".format(output_dir, patientname, label, layer, outputname))
          
     logging.info(f"Generating XAI maps for method 2: SmoothGrad...")
     smoothed_mprage = torch.empty((n_labels,outputs[0,1].size(0),outputs[0,1].size(1),outputs[0,1].size(2)), device='cuda')
     smoothed_flair = torch.empty((n_labels,outputs[0,1].size(0),outputs[0,1].size(1),outputs[0,1].size(2)), device='cuda')
     inputs.requires_grad = False
     
     for q in tqdm(range(nb_smooth)):

          noisy_input = inputs + inputs.new(inputs.size()).normal_(0, std)
          noisy_input.requires_grad_()
          
          outputs = model(noisy_input)
          output_mask = outputs[0,1]

          for label in range(1, n_labels):
               #print("        Lesion {}/{}".format(label,n_labels))
               patch_vector = np.where(wm_labels==label)

               grad_input, = torch.autograd.grad(outputs[0,1][patch_vector].sum(), noisy_input, retain_graph=True)
               
               smoothed_flair[label-1] = smoothed_flair[label-1] + grad_input[0,0]
               smoothed_mprage[label-1] = smoothed_mprage[label-1] + grad_input[0,1]
         
     # Obtain avg over noisy versions and save lesion saliency
     del data
     gc.collect()
     time.sleep(5)

     matrix = (smoothed_flair / (nb_smooth * n_labels)).cpu().numpy()
     
     for label in range(1, n_labels):
          flair_les_saliency = nib.Nifti1Image(matrix[label-1], input_affine)
          nib.save(flair_les_saliency, "{}/{}/flair_smooth_saliency_les_{}_{}".format(output_dir, patientname, label, outputname))
     
     matrix = torch.nn.functional.interpolate((smoothed_mprage / (nb_smooth * n_labels)), mode='bilinear').cpu().numpy()

     for label in range(1, n_labels):
          mprage_les_saliency = nib.Nifti1Image(matrix[label-1], input_affine)
          nib.save(mprage_les_saliency, "{}/{}/mprage_smooth_saliency_les_{}_{}".format(output_dir, patientname, label, outputname))
     logging.info(f"All XAI maps saved for patient ", patientname)
     del matrix, flair_les_saliency, mprage_les_saliency
     gc.collect()
     time.sleep(5)
     i+=1
