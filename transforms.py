from monai.transforms import Compose, CopyItemsd, Lambdad, AddChanneld, Identityd, LoadImaged, NormalizeIntensityd, \
    ConcatItemsd, DeleteItemsd, ToTensord, RandCropByPosNegLabeld, RandSpatialCropd, RandCropByLabelClassesd, \
    RandShiftIntensityd, RandScaleIntensityd, RandFlipd, RandRotate90d, RandAffined
from scipy import ndimage
import numpy as np
import torch


def remove_connected_components(segmentation, l_min=3):
    """Remove small lesions leq than `l_min` voxels from the binary segmentation mask.
    """
    if segmentation.ndim != 3:
        raise ValueError(f"Mask must have 3 dimensions, got {segmentation.ndim}.")
    struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
    labeled_seg, num_labels = ndimage.label(segmentation, structure=struct_el)
    segmentation_tr = np.zeros_like(segmentation)
    for label in range(1, num_labels + 1):
        if np.sum(labeled_seg == label) > l_min:
            segmentation_tr[labeled_seg == label] = 1
    return segmentation_tr


def binarize_mask(prob_map, threshold):
    """Apply threshold to probability mask """
    if isinstance(prob_map, np.ndarray):
        binary_mask = prob_map.copy()
        binary_mask[binary_mask >= threshold] = 1.0
        binary_mask[binary_mask < threshold] = 0.0
    elif isinstance(prob_map, torch.Tensor):
        binary_mask = prob_map.clone()
        binary_mask[binary_mask >= threshold] = 1.0
        binary_mask[binary_mask < threshold] = 0.0
    else:
        raise TypeError(f"Type {type(binarize_mask)} is not supported.")
    return binary_mask


def get_val_transforms(input_keys: list, label_key: str, binarize_keys: list = None, generate_instance_mask: bool = False):
    """
    :param input_keys: mri contrast keys
    :param label_key: target binary mask key
    :param binarize_keys: keys of multi label masks to be binarized
    :return: monai.transforms.Compose instance
    """
    all_keys = input_keys + [label_key]
    all_tr_keys = ["inputs", label_key]
    if generate_instance_mask:
        all_tr_keys += ["instance_mask"]
        geninstm_transform = Compose([
            CopyItemsd(keys=label_key, times=1, names=["instance_mask"]),
            Lambdad(keys="instance_mask",
                    func=lambda x:
                    ndimage.label(x, structure=ndimage.generate_binary_structure(rank=3, connectivity=2))[0].astype(
                        'float32')),
            AddChanneld(keys="instance_mask")
        ])
    else:
        geninstm_transform = Identityd(keys=all_keys)
    if binarize_keys is not None:
        all_keys += binarize_keys
        all_tr_keys += binarize_keys
        bin_transform = Lambdad(keys=binarize_keys, func=lambda x: (x > 0).astype(x.dtype))
    else:
        bin_transform = Identityd(keys=all_keys)
    return Compose(
        [
            LoadImaged(keys=all_keys),
            geninstm_transform,
            AddChanneld(keys=all_keys), bin_transform,
            NormalizeIntensityd(keys=input_keys, nonzero=True, channel_wise=True),
            ConcatItemsd(keys=input_keys, name="inputs"), DeleteItemsd(keys=input_keys),
            ToTensord(keys=all_tr_keys)
        ]
    )

def get_valnotarget_transforms(input_keys: list, binarize_keys: list = None, generate_instance_mask: bool = False):
    """
    :param input_keys: mri contrast keys
    :param label_key: target binary mask key
    :param binarize_keys: keys of multi label masks to be binarized
    :return: monai.transforms.Compose instance
    """
    all_keys = input_keys
    all_tr_keys = ["inputs"]
    geninstm_transform = Identityd(keys=all_keys)
    if binarize_keys is not None:
        all_keys += binarize_keys
        all_tr_keys += binarize_keys
        bin_transform = Lambdad(keys=binarize_keys, func=lambda x: (x > 0).astype(x.dtype))
    else:
        bin_transform = Identityd(keys=all_keys)
    return Compose(
        [
            LoadImaged(keys=all_keys),
            geninstm_transform,
            AddChanneld(keys=all_keys), bin_transform,
            NormalizeIntensityd(keys=input_keys, nonzero=True, channel_wise=True),
            ConcatItemsd(keys=input_keys, name="inputs"), DeleteItemsd(keys=input_keys),
            ToTensord(keys=all_tr_keys)
        ]
    )

def get_train_transforms(input_keys: list, label_key: str, binarize_keys: list = None,
                         balancing_key: str = None, generate_instance_mask: bool = False,
                         n_classes: int = None, crop_factor: float = 4 / 3,
                         roi_size: tuple = (96, 96, 96), n_patches: int = 32):
    """
    Transforms specific to keys:
    Input keys:
        * Intensity normalisation
        * intensity augmentation
        * Concatenation into one image and removal of initial keys
        * Subvolumes generation
        * General augmentation
    Target key:
        * Subvolumes formation
        * General augmentation
    Binarise keys:
        * Mask binarisation
        * Subvolumes formation
        * General augmentation
    Balance keys:
        * Used only for forming balanced dataset based on the amount of classes
    :param n_classes: number of classes including bg, needed when balancing the dataset
    :param generate_instance_mask: if True, will generate instance segmentation targets mask, needed for blobloss
    :param crop_factor: the spatial size after the first crop
    :param input_keys: mri contrast keys
    :param label_key: target binary mask key
    :param binarize_keys: keys of multi label masks to be binarized
    :param balancing_key: key of multi label masks to be used to balance sub-volumes
    :param roi_size: size of patches
    :param n_patches: number of patches
    :return: monai.transforms.Compose instance
    """
    all_keys = input_keys + [label_key]     # will go all necessary transforms like loading, add channel, etc
    all_tr_keys = ["inputs", label_key]           # will also undergo patches formation, augmentation, to tensor
    interp_mode = ["bilinear", 'nearest']        # for rand affine transformation

    first_crop_size = (int(roi_size[0] * crop_factor),
                       int(roi_size[1] * crop_factor),
                       int(roi_size[0] * crop_factor))

    if generate_instance_mask:
        all_tr_keys += ["instance_mask"]
        interp_mode += ['nearest']
        geninstm_transform = Compose([
            CopyItemsd(keys=label_key, times=1, names=["instance_mask"]),
            Lambdad(keys="instance_mask",
                    func=lambda x: ndimage.label(x, structure=ndimage.generate_binary_structure(rank=3, connectivity=2))[0].astype('float32')),
            AddChanneld(keys="instance_mask")
        ])
    else:
        geninstm_transform = Identityd(keys=all_keys)

    if binarize_keys is not None:
        all_keys += binarize_keys
        all_tr_keys += binarize_keys
        interp_mode += ['nearest' for _ in binarize_keys]
        bin_transform = Lambdad(keys=binarize_keys, func=lambda x: (x > 0).astype(x.dtype))
    else:
        bin_transform = Identityd(keys=all_keys)

    if balancing_key is None:
        # case if there is no balancing mask, use label mask to cut 4 times more fg than bg patches
        patch_transforms = Compose([
            RandCropByPosNegLabeld(keys=all_tr_keys, label_key=label_key,
                                   spatial_size=first_crop_size,
                                   pos=4, neg=1, num_samples=n_patches),
            RandSpatialCropd(keys=all_tr_keys, roi_size=roi_size,
                             random_center=True, random_size=False)
        ])
    else:
        all_keys += [balancing_key]
        ratios = [2/n_patches]
        ratios += [(n_patches - 2)/(n_classes - 1)/n_patches for _ in range(1, n_classes)]
        patch_transforms = RandCropByLabelClassesd(keys=all_tr_keys, label_key=balancing_key,
                                                   spatial_size=first_crop_size, ratios=ratios,
                                                   num_classes=n_classes, num_samples=n_patches)
    return Compose(
        [
            # necessary
            LoadImaged(keys=all_keys),
            # if instance mask is to be generated
            geninstm_transform,
            # necessary
            AddChanneld(keys=all_keys), NormalizeIntensityd(keys=input_keys, nonzero=True, channel_wise=True),
            # only if cl mask
            bin_transform,
            # augment intensity
            RandShiftIntensityd(keys=input_keys, offsets=0.1, prob=0.5),
            RandScaleIntensityd(keys=input_keys, factors=0.1, prob=0.5),
            # necessary
            ConcatItemsd(keys=input_keys, name="inputs"),
            DeleteItemsd(keys=input_keys),
            # crop on subvolumes
            patch_transforms,
            # augment subvolumes
            RandFlipd(keys=all_tr_keys, prob=0.5, spatial_axis=(0, 1, 2)),
            RandRotate90d(keys=all_tr_keys, prob=0.5, spatial_axes=(0, 1)),
            RandRotate90d(keys=all_tr_keys, prob=0.5, spatial_axes=(1, 2)),
            RandRotate90d(keys=all_tr_keys, prob=0.5, spatial_axes=(0, 2)),
            RandAffined(keys=all_tr_keys, mode=interp_mode,
                        prob=0.5, spatial_size=roi_size,
                        rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                        scale_range=(0.1, 0.1, 0.1), padding_mode='border'),
            # necessary ?
            ToTensord(keys=all_tr_keys)
        ]
    )
