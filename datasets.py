from monai.data import CacheDataset
from monai.transforms import (
    Lambdad, Compose, LoadImaged)
import os
from glob import glob
import re
import logging
from transforms import binarize_mask


def check_dataset(filepaths_list, prefixes):
    """Check that there are equal amounts of files in both lists and
    the names before prefices are similar for each of the matched pairs of
    flair image filepath and gts filepath.
    Parameters:
        - flair_filepaths (list) - list of paths to flair images
        - gts_filepaths (list) - list of paths to lesion masks of flair images
        - args (argparse.ArgumentParser) - system arguments, should contain `flair_prefix` and `gts_prefix`
    """
    if len(filepaths_list) > 1:
        filepaths_ref = filepaths_list[0]
        for filepaths, prefix in zip(filepaths_list[1:], prefixes[1:]):
            assert len(filepaths_ref) == len(
                filepaths), f"Found {len(filepaths_ref)} ref files and {len(filepaths)} with prefix: {prefix}"
        for sub_filepaths in list(map(list, zip(*filepaths_list))):
            filepath_ref = sub_filepaths[0]
            prefix_ref = prefixes[0]
            for filepath, prefix in zip(sub_filepaths[1:], prefixes[1:]):
                if os.path.basename(filepath_ref)[:-len(prefix_ref)] != os.path.basename(filepath)[:-len(prefix)]:
                    raise ValueError(f"{filepath_ref} and {filepath} do not match")


def get_filepaths(paths, prefixes):
    if isinstance(paths, list):
        return [sorted(glob(os.path.realpath(os.path.join(pa, f"*/*/{pre}")), recursive=True), key=lambda i: int(re.sub('\D', '', i)))
                for pa, pre in zip(paths, prefixes)]      
    elif isinstance(paths, str):
        return sorted(glob(os.path.realpath(os.path.join(paths, f"*/*/{prefixes}")), recursive=True),
                      key=lambda i: int(re.sub('\D', '', i)))
    else:
        return None


class NiftiDataset(CacheDataset):
    def __init__(self, input_paths: list, input_prefixes: list, input_names: list,
                 target_path: str, target_prefix: list, transforms,
                 balmask_path: str = None, balmask_prefix: str = None,
                 num_workers=0, cache_rate=0.5):
        """
        :param input_paths: list of paths to directories where input MR images are stored
        :param target_path: path to the directory where target binary masks are stored
        :param balmask_path: path to the directory where balancing masks are stored
        :param input_prefixes: list of name endings of files in corresponding `input_paths` directories
        :param target_prefix: name endings of files in `target_path` directory
        :param balmask_prefix: name ending of files in `balmask_path` directory
        :param input_names: names of modalities corresponding to files in `input_paths` directories
        :param num_workers: number of parallel processes to preprocess the data
        :param cache_rate: fraction of images that are preprocessed and cached
        """
        if not len(input_paths) == len(input_prefixes) == len(input_names):
            raise ValueError("Input paths, prefixes and names should be of the same length.")

        def get_nonzero_targets(filepaths):
            import nibabel as nib
            return [i_f for i_f, file in enumerate(filepaths)
                    if nib.load(file).get_fdata().sum() > 0.0]

        self.input_filepaths = get_filepaths(input_paths, input_prefixes)
        self.target_filepaths = get_filepaths(target_path, target_prefix)
        self.balmask_filepaths = get_filepaths(balmask_path, balmask_prefix)

        # idx_stay = get_nonzero_targets(self.target_filepaths)
        #
        # self.target_filepaths = [filename for i_f, filename in enumerate(self.target_filepaths)
        #                          if i_f in idx_stay]
        # self.input_filepaths = [filename for i_f, filename in enumerate(self.input_filepaths)
        #                         if i_f in idx_stay]

        to_check_filepaths = self.input_filepaths + [self.target_filepaths]
        to_check_prefix = input_prefixes + [target_prefix]
        modality_names = input_names + ["targets"]
        if self.balmask_filepaths is not None:
            # self.balmask_filepaths = [filename for i_f, filename in enumerate(self.balmask_filepaths)
            #                           if i_f in idx_stay]
            to_check_filepaths += [self.balmask_filepaths]
            to_check_prefix += [balmask_prefix]
            modality_names += ["balance_mask"]

        check_dataset(to_check_filepaths, to_check_prefix)

        logging.info(f"Initializing the dataset. Number of subjects {len(self.target_filepaths)}")

        self.files = [dict(zip(modality_names, files)) for files in list(zip(*to_check_filepaths))]

        super().__init__(data=self.files, transform=transforms,
                         cache_rate=cache_rate, num_workers=num_workers, hash_as_key=True)

    def __len__(self):
        return len(self.files)

class NiftinotargetDataset(CacheDataset):
    def __init__(self, input_paths: list, input_prefixes: list, input_names: list,
                 transforms, balmask_path: str = None, balmask_prefix: str = None,
                 num_workers=0, cache_rate=0.5):
        """
        :param input_paths: list of paths to directories where input MR images are stored
        :param balmask_path: path to the directory where balancing masks are stored
        :param input_prefixes: list of name endings of files in corresponding `input_paths` directories
        :param balmask_prefix: name ending of files in `balmask_path` directory
        :param input_names: names of modalities corresponding to files in `input_paths` directories
        :param num_workers: number of parallel processes to preprocess the data
        :param cache_rate: fraction of images that are preprocessed and cached
        """
        if not len(input_paths) == len(input_prefixes) == len(input_names):
            raise ValueError("Input paths, prefixes and names should be of the same length.")

        self.input_filepaths = get_filepaths(input_paths, input_prefixes)
        self.balmask_filepaths = get_filepaths(balmask_path, balmask_prefix)

        to_check_filepaths = self.input_filepaths
        to_check_prefix = input_prefixes
        modality_names = input_names
 
        check_dataset(to_check_filepaths, to_check_prefix)

        logging.info(f"Initializing the dataset. Number of subjects {len(list(zip(*to_check_filepaths)))}")

        self.files = [dict(zip(modality_names, files)) for files in list(zip(*to_check_filepaths))]

        super().__init__(data=self.files, transform=transforms,
                         cache_rate=cache_rate, num_workers=num_workers, hash_as_key=True)

    def __len__(self):
        return len(self.files)

class NiftiCLWMLDataset(CacheDataset):
    def __init__(self, input_paths: list, input_prefixes: list, input_names: list,
                 target_path: str, target_prefix: list, transforms,
                 clmask_path: str, clmask_prefix: str, wmlmask_path: str, wmlmask_prefix: str,
                 num_workers=0, cache_rate=0.5):
        """ Made for evaluation separated between cl and wml
        :param input_paths: list of paths to directories where input MR images are stored
        :param target_path: path to the directory where target binary masks are stored
        :param input_prefixes: list of name endings of files in corresponding `input_paths` directories
        :param target_prefix: name endings of files in `target_path` directory
        :param input_names: names of modalities corresponding to files in `input_paths` directories
        :param num_workers: number of parallel processes to preprocess the data
        :param cache_rate: fraction of images that are preprocessed and cached
        """
        if not len(input_paths) == len(input_prefixes) == len(input_names):
            raise ValueError("Input paths, prefixes and names should be of the same length.")

        self.input_filepaths = get_filepaths(input_paths, input_prefixes)
        self.target_filepaths = get_filepaths(target_path, target_prefix)
        self.clmask_filepaths = get_filepaths(clmask_path, clmask_prefix)
        self.wmlmask_filepaths = get_filepaths(wmlmask_path, wmlmask_prefix)

        to_check_filepaths = self.input_filepaths + [self.target_filepaths, self.clmask_filepaths,
                                                     self.wmlmask_filepaths]
        to_check_prefix = input_prefixes + [target_prefix, clmask_prefix, wmlmask_prefix]
        modality_names = input_names + ["targets", "targets_cl", "targets_wml"]
        check_dataset(to_check_filepaths, to_check_prefix)

        logging.info(f"Initializing the dataset. Number of subjects {len(self.target_filepaths)}")

        self.files = [dict(zip(modality_names, files)) for files in list(zip(*to_check_filepaths))]

        super().__init__(data=self.files, transform=transforms,
                         cache_rate=cache_rate, num_workers=num_workers, hash_as_key=True)

    def __len__(self):
        return len(self.files)


class PredictedCLWMLDataset(CacheDataset):
    def __init__(self, path_pred: str, class_number: int,
                 clmask_path: str, clmask_prefix: str, wmlmask_path: str, wmlmask_prefix: str,
                 num_workers: int = 0, cache_rate: float = 0.5):
        """
        Limitations:
        - For the case when the predictions have already been obtained.
        - For now only works for a single class predictions.
        :param path_pred: path to the directory where target binary masks are stored
        :param class_number: number of the corresponding class
        :param num_workers: number of parallel processes to preprocess the data
        :param cache_rate: fraction of images that are preprocessed and cached
        """
        self.target_filepaths = get_filepaths(path_pred,
                                              f"all_lesions_isovox_target_class_{class_number}.nii.gz")
        self.pred_prob_filepaths = get_filepaths(path_pred,
                                                 f"all_lesions_isovox_pred_prob_class_{class_number}.nii.gz")
        self.clmask_filepaths = get_filepaths(clmask_path, clmask_prefix)
        self.wmlmask_filepaths = get_filepaths(wmlmask_path, wmlmask_prefix)

        to_check_filepaths = [self.target_filepaths, self.pred_prob_filepaths, self.clmask_filepaths,
                              self.wmlmask_filepaths]
        to_check_prefix = [
            f"all_lesions_isovox_target_class_{class_number}.nii.gz",
            f"all_lesions_isovox_pred_prob_class_{class_number}.nii.gz",
            clmask_prefix, wmlmask_prefix
        ]
        modality_names = ["targets", "outputs", "targets_cl", "targets_wml"]
        check_dataset(to_check_filepaths, to_check_prefix)

        logging.info(f"Initializing the dataset. Number of subjects {len(self.target_filepaths)}")

        self.files = [dict(zip(modality_names, files)) for files in list(zip(*to_check_filepaths))]

        self.transform = Compose(
            [
                LoadImaged(keys=["targets", "outputs", "targets_cl", "targets_wml"]),
                Lambdad(keys=["targets_cl", "targets_wml"], func=lambda x: (x > 0).astype(x.dtype))
            ]
        )

        super().__init__(data=self.files, transform=self.transform,
                         cache_rate=cache_rate, num_workers=num_workers, hash_as_key=True)

    def __len__(self):
        return len(self.files)
