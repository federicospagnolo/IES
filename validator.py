import torch
from transforms import binarize_mask
from monai.networks.utils import one_hot
from losses import BlobNormalisedDiceLoss, DetectionLoss


class Validator:
    def __init__(self, data_loader, activation, metrics: dict, device,
                 loss_function, inferer, prob_threshold: float, n_classes: int,
                 include_background: bool = False, to_onehot_y: bool = True,
                 inputs_key: str = "inputs", targets_key: str = "targets", instance_key: str = "instance_mask"):
        self.data_loader = data_loader
        self.activation = activation
        self.metrics = metrics
        self.inferer = inferer
        self.loss_function = loss_function
        self.inputs_key = inputs_key
        self.targets_key = targets_key
        self.instance_key = instance_key
        self.device = device
        self.threshold = prob_threshold
        self.include_background = include_background
        self.n_classes = n_classes
        self.to_onehot_y = to_onehot_y
        self.send_instmask = isinstance(loss_function, (BlobNormalisedDiceLoss, DetectionLoss))

    def __call__(self, network, *args, **kwargs):
        network.eval()
        metrics_names = [m + '_' + str(c) for m in self.metrics.keys()
                         for c in range(0 if self.include_background else 1, self.n_classes)] + ['loss']
        metrics_val = dict(zip(metrics_names, [0.0 for _ in metrics_names]))
        count = 0
        with torch.no_grad():
            for data in self.data_loader:
                count += 1
                inputs, targets = data[self.inputs_key].to(self.device), data[self.targets_key].to(self.device)
                outputs = self.inferer(inputs=inputs, network=network)  # [1, 2, H, W, D]
                outputs = self.activation(outputs)  # [1, 2, H, W, D]

                if self.send_instmask:
                    instance_mask = data[self.instance_key].to(self.device)
                    metrics_val['loss'] += self.loss_function(outputs, targets, instance_mask).cpu().item()
                else:
                    metrics_val['loss'] += self.loss_function(outputs, targets).cpu().item()

                if self.to_onehot_y:
                    targets = one_hot(targets, num_classes=self.n_classes)

                outputs = outputs.squeeze(0).cpu().numpy()  # [2, H, W, D]
                targets = targets.squeeze(0).cpu().numpy()  # [2, H, W, D]

                outputs = binarize_mask(prob_map=outputs, threshold=self.threshold)
                for c in range(0 if self.include_background else 1, self.n_classes):
                    metrics_val_c = dict()
                    for metric_func in self.metrics.values():
                        metrics_val_c.update(metric_func(y_pred=outputs[c], y=targets[c], check=True))
                    for metric_name in metrics_names:
                        if metric_name[-len(f'_{c}'):] == f'_{c}':
                            metrics_val[metric_name] += metrics_val_c[metric_name.split('_')[0]]

        for metric_name, metric_val in metrics_val.items():
            metrics_val[metric_name] /= count

        return metrics_val
