################
#written by Federico Spagnolo
#usage: python ies_maxpool_nets_fix.py --model_checkpoint ../WMLs_NM_FS_gpu/SMSC_nn-unet_train/model_epoch_31.pth --input_val_paths ../SMSC/test ../SMSC/test --input_prefixes flair_3d_sbr.nii.gz t1n_3d_sb.nii.gz --target_val_path ../SMSC/test --target_prefix lesion_mask_final.nii.gz --num_workers 0 --cache_rate 0.01 --threshold 0.4 --model nn-unet

#python ies_maxpool_nets_fix.py --model_checkpoint ../WMLs_NM_FS_gpu/SMSC_swin_unetr_train/model_epoch_41.pth --input_val_paths ../SMSC/test ../SMSC/test --input_prefixes flair_3d_sbr.nii.gz t1n_3d_sb.nii.gz --target_val_path ../SMSC/test --target_prefix lesion_mask_final.nii.gz --num_workers 0 --cache_rate 0.01 --threshold 0.4 --model swin_unetr --use_checkpoint
################
# Import torch
import torch
import torchvision
#import packages for model description
from torchvision import models
import scipy.ndimage as ndimage
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import UNet, SwinUNETR, UNETR, DynUNet
from monai.transforms import (Activations, AddChanneld, Compose, ConcatItemsd,
                              LoadImaged, NormalizeIntensityd, RandAffined,
                              RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
                              RandShiftIntensityd, RandSpatialCropd, Spacingd, SelectItemsd,
                              ToTensord)
from monai.networks.nets import UNet
from monai.visualize import GradCAMpp
from datasets import NiftinotargetDataset, NiftiDataset
from transforms import remove_connected_components, get_val_transforms, get_valnotarget_transforms, binarize_mask
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
from torch.utils.checkpoint import checkpoint

def checkpointed_model(x):
    # MONAI inferer expects network(input), so wrap model in a checkpoint
    return checkpoint(model, x)

def dp(path1: str, path2: str) -> str:
    return os.path.join(path1, path2)


# current folder
script_folder = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(
description='''Saliency map generation.
                        If no arguments are given, all the default values will be used.''', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='-d', dest='data_path', default=script_folder, help="Path where the data is")
parser.add_argument('--model', type=str, default='unet-', help="unet-")
parser.add_argument('--model_checkpoint', metavar='-ckpt', dest='model_checkpoint', default="model_epoch_1.pth", help="Path to he best checkpoint of the model")
parser.add_argument('--input_val_paths', type=str, nargs='+', required=True)
parser.add_argument('--input_prefixes', type=str, nargs='+', required=True)
parser.add_argument('--target_val_path', type=str)
parser.add_argument('--target_prefix', type=str)
# data handling
parser.add_argument('--input_size', type=float, default=96, help="size of the patch")
parser.add_argument('--num_workers', type=int, default=0,
                        help='number of jobs used to load the data, DataLoader parameter')
parser.add_argument('--cache_rate', type=float, default=0.1, help='cache dataset parameter')
parser.add_argument('--threshold', type=float, default=0.5, help='lesion threshold')
parser.add_argument('--use_checkpoint', action='store_true', help='Recomputes activations from forward pass each time instead of storing. Use to save GPU memory at the expense of computation speed.')
args = parser.parse_args()        

data_path = args.data_path
model_checkpoint = args.model_checkpoint
args.input_modalities = ['flair', 'mprage']
args.n_classes = 2
seed = 1


# set output folder
group = "TP"
output_dir = f"attention_maps_{args.model}/" + group
os.makedirs(output_dir, exist_ok=True)

''' Get default device '''
logging.basicConfig(level = logging.INFO)
print("total devices", torch.cuda.device_count())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logging.info(f"Using device: {device}")
torch.multiprocessing.set_sharing_strategy('file_system')

# Init your model
if args.model == 'unet-':
        model = UNet(spatial_dims=3, in_channels=len(args.input_modalities), out_channels=args.n_classes,
                     channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), norm='batch', num_res_units=0).to(device)
        # Weights intialization
        for layer in model.model.modules():
                if type(layer) == torch.nn.Conv3d:
                        torch.nn.init.xavier_normal_(layer.weight, gain=1.0)             
elif args.model == "unetr":
        model = UNETR(img_size=args.input_size, in_channels=len(args.input_modalities),
                      out_channels=args.n_classes, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, dropout_rate=0.1).to(device)
elif args.model == 'swin_unetr':
        model = SwinUNETR(img_size=args.input_size, in_channels=len(args.input_modalities),
                          out_channels=args.n_classes).to(device)
elif args.model == 'nn-unet':
        model = DynUNet(spatial_dims=3, in_channels=len(args.input_modalities), out_channels=args.n_classes, kernel_size=[3, 3, 3], strides=[1, 2, 2], upsample_kernel_size=[2, 2], filters=[16, 32, 64], dropout=0.1, res_block=True, norm_name="instance", act_name="leakyrelu").to(device)


# Load best model weights
model_path = dp(script_folder, model_checkpoint)
print(model_path)
if args.model == 'unet-':
     model = torch.nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
model.eval()
activation = torch.nn.Softmax(dim=1)

''' Define validation actors '''
inferer = SlidingWindowInferer(roi_size=(96, 96, 96),
                                   sw_batch_size=1, mode='gaussian', overlap=0.25)

logging.info(f"Best model save file loaded from: {model_path}")

# Load Dataset
val_transforms = get_val_transforms(input_keys=args.input_modalities, label_key="targets").set_random_state(seed=seed)
val_dataset = NiftiDataset(input_paths=args.input_val_paths, input_prefixes=args.input_prefixes,
                                   input_names=args.input_modalities,
                                   target_path=args.target_val_path,  target_prefix=args.target_prefix, 
                                   transforms=val_transforms, num_workers=args.num_workers,
                                   cache_rate=args.cache_rate)
#val_transforms = get_valnotarget_transforms(input_keys=args.input_modalities).set_random_state(seed=seed)
#val_dataset = NiftinotargetDataset(input_paths=args.input_val_paths, input_prefixes=args.input_prefixes, input_names=args.input_modalities, transforms=val_transforms, num_workers=args.num_workers, cache_rate=args.cache_rate)                                                                      
                                   
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=args.num_workers)

i = 0
threshold = args.threshold
std_fraction = 0.05
nb_smooth = 50

#layer = "module.model.1.submodule.2.conv"
#for name, _ in model.named_modules(): print(name)
#cam = GradCAMpp(nn_module=model, target_layers=layer)
logging.info(f"Initializing the dataset. Number of subjects {len(val_dataloader)}")

# Batch Loading

for data in val_dataloader:

     filename = data['flair_meta_dict']['filename_or_obj'][0]
     logging.info(f"Evaluate gradients in batch {filename}")
     
     with torch.no_grad():
         inputs = data["inputs"].to(device) # 0 is flair, 1 is mprage

         #outputs = model(inputs)
         outputs = activation(inferer(inputs=inputs, network=model))  # [1, 2, H, W, D]

     #outputs = activation(outputs)  # [1, 2, H, W, D]
     output_mask = outputs[0,1].detach().cpu().numpy()
     output_mask[output_mask > threshold] = 1
     output_mask[output_mask < threshold] = 0
     
     std = std_fraction * (inputs[0,0].max() - inputs[0,0].min())
     input_affine = nib.load(filename).affine
     
     # Save predicted output
     wm_lesion_mask = np.array(data["targets"].squeeze())
     wm_lesion_mask = binarize_mask(wm_lesion_mask, 1.)
     pred = nib.Nifti1Image(output_mask, input_affine)
     outputname = (filename.split("test/", 1)[1]).split("/flair", 1)[0].replace('/', '-') + ".nii.gz"
     patientname = (filename.split("test/", 1)[1]).split("/flair", 1)[0].replace('/', '-')
     
     try:
            os.mkdir('./' + output_dir + '/' + patientname)
     except FileExistsError:
            logging.info(f"Directory {'./' + output_dir + '/' + patientname} already exists")

     nib.save(pred, "{}/{}/pred_{}".format(output_dir, patientname, outputname))
     
     wm_labels = get_lesion_types_masks(output_mask, wm_lesion_mask, 'non_zero', n_jobs = 2)['TPL'] #TP
     #wm_labels = get_lesion_types_masks(output_mask, wm_lesion_mask, 'non_zero', n_jobs = 2)['FPL'] #FP
     #wm_labels = get_lesion_types_masks(output_mask, wm_lesion_mask, 'non_zero', n_jobs = 2)['FNL'] #FN
     
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
          
     logging.info(f"Generating XAI maps for method 2: SmoothGrad...")
     smoothed_mprage = torch.zeros((n_labels,inputs[0,1].size(0),inputs[0,1].size(1),inputs[0,1].size(2)), device=device)
     smoothed_flair = torch.zeros((n_labels,inputs[0,1].size(0),inputs[0,1].size(1),inputs[0,1].size(2)), device=device)
     
     for q in tqdm(range(nb_smooth)):

         noisy_input = inputs + inputs.new(inputs.size()).normal_(0, std)
         noisy_input.requires_grad_()
         
         if args.use_checkpoint:
               outputs = inferer(inputs=noisy_input, network=checkpointed_model)
         else:
               outputs = inferer(inputs=noisy_input, network=model)

         for label in range(34, 35):
               print("        Lesion {}/{}".format(label,n_labels))
               patch_vector = np.where(wm_labels==label)
               if len(patch_vector[0]) == 0:
                    continue
               
               gcomp = torch.zeros((2,outputs[0,1].size(0),outputs[0,1].size(1),outputs[0,1].size(2)), device=device)

               # Convert lesion voxel indices to tensor
               vox_idx = torch.as_tensor(np.vstack(patch_vector).T, device=device, dtype=torch.long)

               # Process lesion voxels in batches to save time
               chunk_size = 512  # adjust depending on GPU memory 512
               n_vox = vox_idx.size(0)
               
               for start in range(0, n_vox, chunk_size):
                    end = min(start + chunk_size, n_vox)
                    sub_vox = vox_idx[start:end]
                    print("        Voxels {}–{}/{}".format(start+1, end, n_vox))
                    
                    # Select model outputs for this voxel batch
                    target = outputs[0, 1, sub_vox[:,0], sub_vox[:,1], sub_vox[:,2]]

                    # Compute combined gradients for this batch
                    noisy_input.grad = None  # reset gradients
                    target.sum().backward(retain_graph=True)
                    grad_input = noisy_input.grad
                    
                    # Elementwise max of absolute gradients (same logic as before)
                    gcomp = torch.where(torch.abs(grad_input) > torch.abs(gcomp), grad_input, gcomp)

               smoothed_flair[label-1] += gcomp[0,0]
               smoothed_mprage[label-1] += gcomp[0,1]
                    
     # Obtain avg over noisy versions and save lesion saliency
     matrix_fl = (smoothed_flair / nb_smooth).cpu().numpy()
     
     del data, noisy_input
     
     for label in range(34, 35):
          flair_les_saliency = nib.Nifti1Image(matrix_fl[label-1], input_affine)
          nib.save(flair_les_saliency, "{}/{}/flair_smooth_saliency_les_{}_{}".format(output_dir, patientname, label, outputname))
     
     matrix_mp = (smoothed_mprage / nb_smooth).cpu().numpy()

     for label in range(34, 35):
          mprage_les_saliency = nib.Nifti1Image(matrix_mp[label-1], input_affine)
          nib.save(mprage_les_saliency, "{}/{}/mprage_smooth_saliency_les_{}_{}".format(output_dir, patientname, label, outputname))
     logging.info(f"All XAI maps saved for patient {patientname}")
     
     i+=1        
