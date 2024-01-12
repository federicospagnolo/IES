################
#written by Federico Spagnolo
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
output_dir = "attention_maps/output"

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
                     channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), norm='batch', num_res_units=0).to(device)
# Weights intialization
for layer in model.model.modules():
        if type(layer) == torch.nn.Conv3d:
                torch.nn.init.xavier_normal_(layer.weight, gain=1.0)

# Load best model weights
model_path = dp(script_folder, model_checkpoint)
print(model_path)
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load(args.model_checkpoint, map_location='cuda'))
model.eval()
activation = torch.nn.Softmax(dim=1)

''' Define validation actors '''
inferer = SlidingWindowInferer(roi_size=(96, 96, 96),
                                   sw_batch_size=4, mode='gaussian', overlap=0.25)

print("Best model save file loaded from:", model_path)

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
     print("Evaluate gradients in batch", filename)
     inputs = data["inputs"].to(device) # 0 is flair, 1 is mprage
     std = std_fraction * (inputs[0,0].max() - inputs[0,0].min())
     input_affine = nib.load(filename).affine
     
     #outputs = model(inputs)
     inputs.requires_grad_()
     outputs = inferer(inputs=inputs, network=model)  # [1, 2, H, W, D]
     outputs = activation(outputs)  # [1, 2, H, W, D]
     grad_input, = torch.autograd.grad(outputs[0,1].sum(), inputs)
     flair_grad3d = grad_input[0,0].detach()
     flair_grad3d[flair_grad3d != 0] = 1
     print(flair_grad3d.sum())
     break
     output_mask = outputs[0,1].detach().cpu().numpy()
     output_mask[output_mask > threshold] = 1
     output_mask[output_mask < threshold] = 0

     # Save predicted output
     pred = nib.Nifti1Image(output_mask, input_affine)
     outputname = (filename.split("ectrims/", 1)[1]).split("/flair", 1)[0].replace('/', '-') + ".nii.gz"
     nib.save(pred, "{}/pred_{}".format(output_dir, outputname))
     
     wm_labels = get_lesion_types_masks(output_mask, output_mask, 'non_zero', n_jobs = 2)['TPL']        
     selected_group = nib.Nifti1Image(wm_labels, input_affine)

     print("unpruned are: ", np.max(wm_labels))
     label = 1
     while label <= np.max(wm_labels):
          patch_vector = np.where(wm_labels==label)
          if len(patch_vector[0])<=5:
               wm_labels[wm_labels==label] = 0
               wm_labels[wm_labels>label] = wm_labels[wm_labels>label] - 1
          else:
               label = label + 1     

     n_labels = np.max(wm_labels)
     print("pruned are: ", n_labels)
     
     selected_group = nib.Nifti1Image(wm_labels, input_affine)
     nib.save(selected_group, "{}/group_{}".format(output_dir, outputname))
     
     smoothed_mprageq = [0]
     smoothed_flairq = [0]
          
     for q in tqdm(range(nb_smooth)):
          noisy_input = inputs + inputs.new(inputs.size()).normal_(0, std)
          noisy_input.requires_grad_()
          outputs = model(noisy_input)
          output_mask = outputs[0,1].detach().cpu().numpy()
          smoothed_mprage = []
          smoothed_flair = []
          
          for label in range(1, n_labels):
               #print("        Lesion {}/{}".format(label,n_labels))
               patch_vector = np.where(wm_labels==label)

               flair_grad3d = np.zeros_like(output_mask)
               mprage_grad3d = np.zeros_like(output_mask)
               grad_input, = torch.autograd.grad(outputs[0,1][patch_vector].sum(), noisy_input, retain_graph=True)
               flair_grad3d = grad_input[0,0].detach().cpu().numpy()
               mprage_grad3d = grad_input[0,1].detach().cpu().numpy()

               smoothed_flair.insert(label-1, flair_grad3d/len(patch_vector[0]))
               smoothed_mprage.insert(label-1, mprage_grad3d/len(patch_vector[0]))
               #print(np.shape(smoothed_mprage))
          # Sum over noisy versions     
          smoothed_flairq = np.add(np.array(smoothed_flairq), np.array(smoothed_flair))
          smoothed_mprageq = np.add(np.array(smoothed_mprageq), np.array(smoothed_mprage))
          #print(np.shape(smoothed_mprageq))           
     # Obtain avg over noisy versions and save lesion saliency
     #print(np.shape(smoothed_flairq))
     flair = smoothed_flairq / (nb_smooth)
     mprage = smoothed_mprageq / (nb_smooth)
     
     flair_les = np.zeros_like(output_mask)
     mprage_les = np.zeros_like(output_mask)
     
     for label in range(1, n_labels):
          #print(np.shape(flair_les))
          patch_vector = np.where(wm_labels==label)
          gcam = cam(x=inputs, class_idx=1, footprint=patch_vector)
          gcam_map = nib.Nifti1Image(gcam[0,0].cpu().numpy(), input_affine)
          nib.save(gcam_map, "{}/GradCAM_lesion{}_layer_{}_{}".format(output_dir, label, layer, outputname))
          
          flair_les = flair[label-1]
          mprage_les = mprage[label-1]
          flair_les_saliency = nib.Nifti1Image(flair_les, input_affine)
          mprage_les_saliency = nib.Nifti1Image(mprage_les, input_affine)
          nib.save(flair_les_saliency, "{}/flair_smooth_saliency_les_{}_{}".format(output_dir, label, outputname))
          nib.save(mprage_les_saliency, "{}/mprage_smooth_saliency_les_{}_{}".format(output_dir, label, outputname))   
     i+=1            
