import logging
import os
import torch
from transforms import binarize_mask
from monai.networks.utils import one_hot
from monai.data import write_nifti
import numpy as np
import pandas as pd


class Evaluator:
    def __init__(self, data_loader, activation, metrics: list, device, inferer, prob_threshold: float, n_classes: int,
                 save_path: str, set_name:str, save_pred: bool = False,
                 include_background: bool = False, to_onehot_y: bool = True, postprocessing=None,
                 inputs_key: str = "inputs", targets_key: str = "targets"):
        self.data_loader = data_loader
        self.activation = activation
        self.metrics = metrics
        self.inferer = inferer
        self.postprocessing = postprocessing
        self.inputs_key = inputs_key
        self.targets_key = targets_key
        self.device = device
        self.threshold = prob_threshold
        self.include_background = include_background
        self.n_classes = n_classes
        self.to_onehot_y = to_onehot_y
        self.save_pred = save_pred

        if self.metrics:
            self.res_filepath = os.path.join(save_path, f'{set_name}_metrics.csv')

        self.save_path_pred = os.path.join(save_path, f"predictions_{set_name}")
        os.makedirs(self.save_path_pred, exist_ok=True)

        if postprocessing is None:
            self.postprocessing = lambda x: x

    def __call__(self, network, *args, **kwargs):
        network.eval()
        metrics_list = []
        filenames = []
        count = 0
        with torch.no_grad():
            for data in self.data_loader:
                count += 1
                inputs, targets = data[self.inputs_key].to(self.device), data[self.targets_key].to(self.device)
                outputs = self.inferer(inputs=inputs, network=network)  # [1, 2, H, W, D]
                outputs = self.activation(outputs)  # [1, 2, H, W, D]

                if self.to_onehot_y:
                    targets = one_hot(targets, num_classes=self.n_classes)

                outputs = outputs.squeeze(0).cpu().numpy()  # [2, H, W, D]
                targets = targets.squeeze(0).cpu().numpy()  # [2, H, W, D]

                outputs_bin = binarize_mask(prob_map=outputs, threshold=self.threshold)
                if self.metrics:
                    for c in range(0, self.n_classes):
                        outputs_bin[c] = self.postprocessing(outputs_bin[c])
                    metrics_val = dict()
                    for c in range(0 if self.include_background else 1, self.n_classes):
                        for metric_func in self.metrics:
                            metrics_row = metric_func(y_pred=outputs_bin[c], y=targets[c], check=True)
                            for key in metrics_row:
                                metrics_val['%s_%d' % (key, c)] = metrics_row[key]

                    metrics_list += [metrics_val]
                    
                filename = data['targets_meta_dict']['filename_or_obj'][0]
                filename = (filename.split("test/", 1)[1]).split("/lesion", 1)[0].replace('/', '-')
                filenames += [filename]

                logging.info(filename)

                if self.metrics:
                    pd.DataFrame(metrics_list, index=filenames).to_csv(self.res_filepath)

                if self.save_pred:
                    affine = data['targets_meta_dict']['affine'][0]
                    spatial_shape = data['targets_meta_dict']['spatial_shape'][0]

                    for c in range(0 if self.include_background else 1, self.n_classes):
                        ''' Save binary mask '''
                        new_filepath = os.path.join(self.save_path_pred, filename.split('.')[0] + f'_pred_class_{c}.nii.gz')
                        write_nifti(outputs_bin[c], new_filepath, affine=affine,
                                    target_affine=affine, output_spatial_shape=spatial_shape)

                        ''' Save probability mask '''
                        new_filepath = os.path.join(self.save_path_pred, filename.split('.')[0] + f'_pred_prob_class_{c}.nii.gz')
                        write_nifti(outputs[c], new_filepath, affine=affine,
                                    target_affine=affine, output_spatial_shape=spatial_shape)

                        new_filepath = os.path.join(self.save_path_pred,
                                                    filename.split('.')[0] + f'_target_class_{c}.nii.gz')
                        write_nifti(targets[c], new_filepath, affine=affine,
                                    target_affine=affine, output_spatial_shape=spatial_shape)

        return metrics_list, filenames


class EvaluatorCLWML:
    def __init__(self, data_loader, activation, metrics: list, cl_wml_metrics: list, device, inferer, prob_threshold: float, n_classes: int,
                 include_background: bool = False, to_onehot_y: bool = True, postprocessing=None, save_path: str = None,
                 inputs_key: str = "inputs", targets_key: str = "targets", cl_key: str = "targets_cl", wml_key: str = "targets_wml"):
        """
        :param data_loader:
        :param activation:
        :param metrics:
        :param device:
        :param inferer:
        :param prob_threshold:
        :param n_classes:
        :param include_background:
        :param to_onehot_y:
        :param postprocessing:
        :param save_path:
        :param inputs_key:
        :param targets_key:
        """
        self.data_loader = data_loader
        self.activation = activation
        self.metrics = metrics
        self.cl_wml_metrics = cl_wml_metrics
        self.inferer = inferer
        self.postprocessing = postprocessing
        self.inputs_key = inputs_key
        self.targets_key = targets_key
        self.cl_key = cl_key
        self.wml_key = wml_key
        self.device = device
        self.threshold = prob_threshold
        self.include_background = include_background
        self.n_classes = n_classes
        self.to_onehot_y = to_onehot_y
        self.save_path = save_path

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

        if postprocessing is None:
            self.postprocessing = lambda x: x

    def __call__(self, network, *args, **kwargs):
        network.eval()
        metrics_list = []
        filenames = []
        count = 0
        with torch.no_grad():
            for data in self.data_loader:
                count += 1
                inputs, targets, targets_cl, targets_wml = \
                    data[self.inputs_key].to(self.device), \
                    data[self.targets_key].to(self.device), \
                    data[self.cl_key].to(self.device), \
                    data[self.wml_key].to(self.device)
                outputs = self.inferer(inputs=inputs, network=network)  # [1, 2, H, W, D]
                outputs = self.activation(outputs)  # [1, 2, H, W, D]

                if self.to_onehot_y:
                    targets = one_hot(targets, num_classes=self.n_classes)
                    targets_cl = one_hot(targets_cl, num_classes=self.n_classes)
                    targets_wml = one_hot(targets_wml, num_classes=self.n_classes)

                outputs = outputs.squeeze(0).cpu().numpy()  # [2, H, W, D]
                targets = targets.squeeze(0).cpu().numpy()  # [2, H, W, D]
                targets_cl = targets_cl.squeeze(0).cpu().numpy()  # [2, H, W, D]
                targets_wml = targets_wml.squeeze(0).cpu().numpy()  # [2, H, W, D]

                outputs_bin = binarize_mask(prob_map=outputs, threshold=self.threshold)
                for c in range(0, self.n_classes):
                    outputs_bin[c] = self.postprocessing(outputs_bin[c])
                metrics_val = dict()
                for c in range(0 if self.include_background else 1, self.n_classes):
                    for metric_func in self.metrics:
                        metrics_row = metric_func(y_pred=outputs_bin[c], y=targets[c], check=True)
                        for key in metrics_row:
                            metrics_val['%s_%d' % (key, c)] = metrics_row[key]
                    for metric_func in self.cl_wml_metrics:
                        metrics_row = metric_func(y_pred=outputs_bin[c], y=targets[c],
                                                  cl_mask=targets_cl[c], wml_mask=targets_wml[c], check=True)
                        for key in metrics_row:
                            metrics_val['%s_%d' % (key, c)] = metrics_row[key]
                    metrics_val['cl_%d' % c] = int(np.sum(targets_cl[c]) > 0.0)
                    metrics_val['wml_%d' % c] = int(np.sum(targets_wml[c]) > 0.0)

                metrics_list += [metrics_val]

                filename = data['targets_meta_dict']['filename_or_obj'][0]
                filename = os.path.basename(filename)
                filenames += [filename]

                logging.info(filename)

                if self.save_path is not None:
                    affine = data['targets_meta_dict']['affine'][0]
                    spatial_shape = data['targets_meta_dict']['spatial_shape'][0]

                    for c in range(0 if self.include_background else 1, self.n_classes):
                        ''' Save binary mask '''
                        new_filepath = os.path.join(self.save_path, filename.split('.')[0] + f'_pred_class_{c}.nii.gz')
                        write_nifti(outputs_bin[c], new_filepath, affine=affine,
                                    target_affine=affine, output_spatial_shape=spatial_shape)

                        ''' Save probability mask '''
                        new_filepath = os.path.join(self.save_path, filename.split('.')[0] + f'_pred_prob_class_{c}.nii.gz')
                        write_nifti(outputs[c], new_filepath, affine=affine,
                                    target_affine=affine, output_spatial_shape=spatial_shape)

                        new_filepath = os.path.join(self.save_path,
                                                    filename.split('.')[0] + f'_target_class_{c}.nii.gz')
                        write_nifti(targets[c], new_filepath, affine=affine,
                                    target_affine=affine, output_spatial_shape=spatial_shape)

        return metrics_list, filenames


class PureEvaluator:
    def __init__(self, data_loader, metrics: list, prob_threshold: float, class_number: int,
                 postprocessing=None):
        """ When the predictions have already been made and saved and are returned by the dataloader
        Works for binary classification only.
        """
        self.data_loader = data_loader
        self.metrics = metrics
        self.postprocessing = postprocessing
        self.outputs_key = "outputs"
        self.targets_key = "targets"
        self.cl_key = "targets_cl"
        self.wml_key = "targets_wml"
        self.threshold = prob_threshold
        self.class_num = class_number

        if postprocessing is None:
            self.postprocessing = lambda x: x

        self.metrics_list = []
        self.filenames = []

    def __call__(self, *args, **kwargs):
        count = 0
        with torch.no_grad():
            for data in self.data_loader:
                count += 1
                outputs, targets = data[self.outputs_key], data[self.targets_key]

                outputs = outputs.squeeze(0).numpy()  # [H, W, D]
                targets = targets.squeeze(0).numpy()

                outputs_bin = binarize_mask(prob_map=outputs, threshold=self.threshold)
                outputs_bin = self.postprocessing(outputs_bin)

                filename = data['targets_meta_dict']['filename_or_obj'][0]
                filename = os.path.basename(filename)
                self.filenames += [filename]

                try:
                    metrics_val = dict()
                    for metric_func in self.metrics:
                        metrics_row = metric_func(y_pred=outputs_bin, y=targets, check=True)
                        for key in metrics_row:
                            metrics_val['%s_%d' % (key, self.class_num)] = metrics_row[key]

                    self.metrics_list += [metrics_val]

                    logging.info(filename)
                except Exception as e:
                    logging.warn(f"Exception caught on {filename}: {e}. Subject excluded from evaluation.")
                    self.filenames.remove(filename)

        return self.metrics_list, self.filenames


class PureEvaluatorCLWML:
    def __init__(self, data_loader, metrics: list, cl_wml_metrics: list, prob_threshold: float, class_number: int,
                 postprocessing=None):
        """ When the predictions have already been made and saved and are returned by the dataloader
        Works for binary classification only.
        """
        self.data_loader = data_loader
        self.metrics = metrics
        self.cl_wml_metrics = cl_wml_metrics
        self.postprocessing = postprocessing
        self.outputs_key = "outputs"
        self.targets_key = "targets"
        self.cl_key = "targets_cl"
        self.wml_key = "targets_wml"
        self.threshold = prob_threshold
        self.class_num = class_number

        if postprocessing is None:
            self.postprocessing = lambda x: x

        self.metrics_list = []
        self.filenames = []

    def __call__(self, *args, **kwargs):
        count = 0
        with torch.no_grad():
            for data in self.data_loader:
                count += 1
                outputs, targets, targets_cl, targets_wml = data[self.outputs_key], data[self.targets_key], \
                                                            data[self.cl_key], data[self.wml_key]   # [1, H, W, D]

                outputs = outputs.squeeze(0).numpy()  # [H, W, D]
                targets = targets.squeeze(0).numpy()
                targets_cl = targets_cl.squeeze(0).numpy()
                targets_wml = targets_wml.squeeze(0).numpy()

                outputs_bin = binarize_mask(prob_map=outputs, threshold=self.threshold)
                outputs_bin = self.postprocessing(outputs_bin)

                filename = data['targets_meta_dict']['filename_or_obj'][0]
                filename = os.path.basename(filename)
                self.filenames += [filename]

                try:
                    metrics_val = dict()
                    for metric_func in self.metrics:
                        metrics_row = metric_func(y_pred=outputs_bin, y=targets, check=True)
                        for key in metrics_row:
                            metrics_val['%s_%d' % (key, self.class_num)] = metrics_row[key]
                    for metric_func in self.cl_wml_metrics:
                        metrics_row = metric_func(y_pred=outputs_bin, y=targets,
                                                  cl_mask=targets_cl, wml_mask=targets_wml, check=True)
                        for key in metrics_row:
                            metrics_val['%s_%d' % (key, self.class_num)] = metrics_row[key]
                    metrics_val['cl_%d' % self.class_num] = int(np.sum(targets_cl) > 0.0)
                    metrics_val['wml_%d' % self.class_num] = int(np.sum(targets_wml) > 0.0)

                    self.metrics_list += [metrics_val]

                    logging.info(filename)
                except Exception as e:
                    logging.warn(f"Exception caught on {filename}: {e}. Subject excluded from evaluation.")
                    self.filenames.remove(filename)

        return self.metrics_list, self.filenames
