# python contextual_info_nets_csv.py --model_checkpoint ../WMLs_NM_FS_gpu/SMSC_nn-unet_train/model_epoch_31.pth --input_val_paths ../SMSC/test ../SMSC/test --target_val_path ../SMSC/test --input_prefixes flair_3d_sbr.nii.gz t1n_3d_sb.nii.gz --target_prefix lesion_mask_final.nii.gz --num_workers 0 --cache_rate 0.01 --threshold 0.4 --model nn-unet
# python contextual_info_nets_csv.py --model_checkpoint ../WMLs_NM_FS_gpu/SMSC_swin_unetr_train/model_epoch_41.pth --input_val_paths ../SMSC/test ../SMSC/test --target_val_path ../SMSC/test --input_prefixes flair_3d_sbr.nii.gz t1n_3d_sb.nii.gz --target_prefix lesion_mask_final.nii.gz --num_workers 0 --cache_rate 0.01 --threshold 0.4 --model swin_unetr
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
from datasets import NiftiDataset
from transforms import remove_connected_components, get_val_transforms, binarize_mask
from losses import *
from lesion_extraction import get_lesion_types_masks
import numpy as np
import nibabel as nib
import pathlib
import argparse
import logging
import os, copy
import sys
from skimage import measure
import matplotlib.pyplot as plt
import numpy.ma as ma
from tqdm import tqdm
import pandas as pd
from plotnine import *
import random

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

parser = argparse.ArgumentParser(
description='''MedCam attention map generation.
                        If no arguments are given, all the default values will be used.''', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='-d', dest='data_path', default=script_folder, help="Path where the data is")
parser.add_argument('--model', type=str, default='unet-', help="unet-")
parser.add_argument('--model_checkpoint', metavar='-ckpt', dest='model_checkpoint', default="model_epoch_1.pth", help="Path to he best checkpoint of the model")
parser.add_argument('--input_val_paths', type=str, nargs='+', required=True)
parser.add_argument('--target_val_path', type=str, required=True)
parser.add_argument('--target_prefix', type=str, required=True)
parser.add_argument('--input_prefixes', type=str, nargs='+', required=True)
# data handling
parser.add_argument('--input_size', type=float, default=96, help="size of the patch")
parser.add_argument('--num_workers', type=int, default=0,
                        help='number of jobs used to load the data, DataLoader parameter')
parser.add_argument('--cache_rate', type=float, default=0.1, help='cache dataset parameter')
parser.add_argument('--threshold', type=float, default=0.5, help='lesion threshold')
args = parser.parse_args()        

# set output folder
output_dir = f"plots_{args.model}"
os.makedirs(output_dir, exist_ok=True)

data_path = args.data_path
model_checkpoint = args.model_checkpoint
args.input_modalities = ['flair', 'mprage']
args.n_classes = 2
seed = 5
random.seed(seed)

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
if args.model == 'unet-':
     model = torch.nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(model_checkpoint, map_location='cuda'))
model.eval()
activation = torch.nn.Softmax(dim=1)

''' Define validation actors '''
inferer = SlidingWindowInferer(roi_size=(96, 96, 96),
                                   sw_batch_size=4, mode='gaussian', overlap=0.25)

print("Best model save file loaded from:", model_path)

# Load Dataset
val_transforms = get_val_transforms(input_keys=args.input_modalities, label_key="targets").set_random_state(seed=seed)
val_dataset = NiftiDataset(input_paths=args.input_val_paths, input_prefixes=args.input_prefixes,
                                   input_names=args.input_modalities,
                                   target_path=args.target_val_path,  target_prefix=args.target_prefix, 
                                   transforms=val_transforms, num_workers=args.num_workers,
                                   cache_rate=args.cache_rate)                                   
                                   
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=args.num_workers)

loss_function = NormalisedDiceFocalLoss(include_background=False, to_onehot_y=True, other_act=activation,
                                                lambda_ndscl=0.5, lambda_focal=1.0)
                                                
threshold = args.threshold
std_fraction = 0.05
score = []
n_detect = []
i = 0

struct1 = np.array(ndimage.generate_binary_structure(3, 2)) # define shape of dilation
#print(struct1)

# Batch Loading
with torch.no_grad():
 for data in val_dataloader:
     filename = data['targets_meta_dict']['filename_or_obj'][0]
     print(i, filename)
 
     print("Evaluate gradients in batch", filename)
     inputs = data["inputs"].to(device) # 0 is flair, 1 is mprage
     std = std_fraction * (inputs[0,0].max() - inputs[0,0].min())
     input_affine = nib.load(val_dataset.target_filepaths[i]).affine
     
     #outputs = model(inputs)    
     outputs = inferer(inputs=inputs, network=model)  # [1, 2, H, W, D]
     outputs = activation(outputs)  # [1, 2, H, W, D]
     output_mask = outputs[0,1].detach().cpu().numpy()
     output_mask[output_mask > threshold] = 1
     output_mask[output_mask < threshold] = 0
     inputs = inputs.cpu()
     
     # Load prediction and annotated mask
     pred_mask = output_mask
     wm_lesion_mask = np.array(data["targets"].squeeze())
     wm_lesion_mask = binarize_mask(wm_lesion_mask, 1.)
     
     # Save predicted output
     pred = nib.Nifti1Image(output_mask, input_affine)
     outputname = (val_dataset.target_filepaths[i].split("test/", 1)[1]).split("/lesion", 1)[0].replace('/', '-') + ".nii.gz"
     nib.save(pred, "{}/pred_{}".format(output_dir, outputname))
     
     wm_labels = get_lesion_types_masks(pred_mask, wm_lesion_mask, 'non_zero', n_jobs = 2)['TPL']
     #wm_labels = get_lesion_types_masks(wm_lesion_mask, pred_mask, 'non_zero', n_jobs = 2)['FNL'] # FP lesions

     print("unpruned are: ", np.max(wm_labels))
     label = 1
     while label <= np.max(wm_labels):
          patch_vector = np.where(wm_labels==label)         
          if (len(patch_vector[0])<=90) | (len(patch_vector[0])>=120):
          #if len(patch_vector[0])<100:
               wm_labels[wm_labels==label] = 0
               wm_labels[wm_labels>label] = wm_labels[wm_labels>label] - 1
          else:
               label = label + 1     

     n_labels = np.max(wm_labels)
     print("pruned are: ", n_labels)
     
     selected_group = nib.Nifti1Image(wm_labels, input_affine)
     nib.save(selected_group, "{}/group_{}".format(output_dir, outputname))
     
     for label in range(1,n_labels+1):
       print("lesion " + str(label))
       
       selected_mask = np.copy(wm_labels)
       selected_mask[selected_mask!=label] = 0.
       selected_mask[selected_mask==label] = 1.
       selected_mask = binarize_mask(selected_mask, 1.)
       SIZE = len(selected_mask[selected_mask==1])

       fake = nib.Nifti1Image(selected_mask, input_affine)
       #nib.save(fake, "{}/selected_mask_{}".format(output_dir, outputname))

       score_tmp = []
       n_detect_tmp = []
       fake_input = torch.zeros_like(inputs)
       
       print("Iteration...")
       for iteration in tqdm(range(1,35,1)): #was 25
          transformed = ndimage.binary_dilation(selected_mask, structure=struct1, iterations=iteration).astype(float)
          fake = nib.Nifti1Image(transformed, input_affine)
          #nib.save(fake, "{}/mask_flair_{}_{}".format(output_dir, iteration, outputname))
          flairtr = inputs[0,0]*transformed
          mpragetr = inputs[0,1]*transformed
          fake = nib.Nifti1Image(flairtr.cpu().numpy(), input_affine)
          #nib.save(fake, "{}/dilated_flair_{}_{}".format(output_dir, iteration, outputname))
     ############
          fake_input[0,0] = flairtr
          fake_input[0,1] = mpragetr
     ############
          #outputs = model(fake_input.to(device))
          outputs = inferer(inputs=fake_input.to(device), network=model)  # [1, 2, H, W, D]
          outputs = activation(outputs)  # [1, 2, H, W, D]
          output_mask = outputs[0,1].detach().cpu().numpy()
          fake = nib.Nifti1Image(output_mask, input_affine)
          #nib.save(fake, "{}/output_{}_{}".format(output_dir, iteration, outputname))
          score_tmp.append((output_mask*selected_mask).sum()/SIZE)
          
          output_mask[output_mask > threshold] = 1
          output_mask[output_mask < threshold] = 0
     
     # Load prediction and annotated mask
          if (output_mask*selected_mask).sum() > 0:
               n_detect_tmp.append(1.)
          else:
               n_detect_tmp.append(0.)
     
       score.append(score_tmp)
       n_detect.append(n_detect_tmp)
     i+=1
                    
avg = []
std = []
N = []

n_detect = np.array(n_detect)
score = np.array(score)
print(score, np.shape(score))

for j in range(0, 34):
    avg.append(np.mean(score[:, j]))
    std.append(np.std(score[:, j]))
    N.append(n_detect[:, j].sum())

# Prepare base data
distances = np.linspace(1, 35, 34)
data_dict = {
    'distance_mm': distances,
    'average_score': avg,
    'std_dev': std,
    'n_predictions': N
}

# Add individual score values as separate columns
# Create one column per sample, e.g. score_0, score_1, ...
for i in range(score.shape[0]):
    data_dict[f'score_{i}'] = score[i, :]

# Convert to DataFrame
df = pd.DataFrame(data_dict)

# Save as CSV
csv_path = os.path.join(output_dir, "contextual_info_data.csv")
df.to_csv(csv_path, index=False)

print(f"Saved lesion analysis data with individual scores to: {csv_path}")
