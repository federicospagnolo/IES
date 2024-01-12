import os.path
import torch
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import pandas as pd
import logging, sys


class EarlyStoppingCallback:
    def __init__(self, patience, tolerance, validator, train_log_file: str,
                 min_is_good: bool = True, metric: str = 'loss'):
        self.patience = patience
        self.tolerance = tolerance
        self.validator = validator
        self.silent_epochs = 0
        self.mult_factor = 1 if min_is_good else -1
        self.best_metric = np.inf if min_is_good else -np.inf
        self.metric = metric

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(train_log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def __call__(self, trainer, *args, **kwargs):
        metric_val = self.validator(trainer.network)[self.metric]
        if self.mult_factor * metric_val > self.mult_factor * self.best_metric or \
                np.abs(metric_val - self.best_metric) < self.tolerance:
            self.silent_epochs += 1
            if self.silent_epochs > self.patience:
                logging.info(f"Early stopping at epoch {trainer.state.epoch + 1}")
                trainer.continue_training = False
        else:
            self.best_metric = metric_val
            self.silent_epochs = 0


class LoggingCallback:
    def __init__(self, logger, validator, print_name, print_freq: int = 1):
        self.validator = validator
        self.logger = logger
        self.print_freq = print_freq
        self.name = print_name

    def __call__(self, trainer, *args, **kwargs):
        if trainer.state.epoch % self.print_freq == 0:
            metrics_dict = self.validator(trainer.network)
            msg = f"{self.name} on epoch {trainer.state.epoch}:"
            for metric_name, metric_val in metrics_dict.items():
                msg += f"\t{metric_name} {metric_val:.5f},"
            self.logger.write(msg[:-1])


class PlottingCallback:
    def __init__(self, logger, metrics_names: list, n_classes: int, include_background: bool = False, print_freq: int = 1):
        self.log_file = logger.log_file
        self.save_path = os.path.dirname(self.log_file)
        self.print_freq = print_freq
        self.n_classes = n_classes - (0 if include_background else 1)
        self.metrics_names = [m + '_' + str(c) for m in metrics_names for c in range(1, self.n_classes + 1)] + ['loss']

    def parse_log_file(self, trainer):
        """ Parse log file """
        with open(self.log_file, 'r') as f:
            df = pd.DataFrame([], columns=["name", "epoch"] + self.metrics_names)
            for line in f.readlines():
                row = {"name": line.split(' ')[0], "epoch": int(re.search(r"epoch [\d]+", line)[0].split(' ')[-1])}
                for metric in self.metrics_names:
                    search_name = re.search(f"{metric} " + r'([+-]?([0-9]*[.])?[0-9]+|nan)', line)[0]
                    row[metric] = float(search_name.split(' ')[1])
                df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
        return df

    def __call__(self, trainer, *args, **kwargs):
        if self.log_file is not None and trainer.state.epoch % self.print_freq == 0:
            df = self.parse_log_file(trainer)
            for metric_name in self.metrics_names:
                for name in df['name'].unique():
                    df_mn = df[df['name'] == name]
                    plt.plot(df_mn['epoch'], df_mn[metric_name], label=name)
                    plt.title(metric_name)
                    plt.xlabel("Epoch")
                    plt.ylabel(metric_name)
                plt.legend()
                plt.savefig(os.path.join(self.save_path, metric_name + '_history.jpg'), dpi=300)
                plt.close()


class SaveModelCallback:
    def __init__(self, save_freq: int, save_path: str, train_log_file: str):
        self.save_freq = save_freq
        self.save_path = save_path
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(train_log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def __call__(self, trainer, *args, **kwargs):
        epoch = trainer.state.epoch if trainer.continue_training else 'last'
        if trainer.state.epoch % self.save_freq == 0 or epoch == 'last':
            torch.save(trainer.network.state_dict(), os.path.join(self.save_path, f"model_epoch_{epoch}.pth"))
            logging.info(f"Saved model at epoch {trainer.state.epoch}")


class PlotLRCallback:
    def __init__(self, logger):
        self.log_file = logger.log_file
        self.filepath = os.path.join(os.path.dirname(self.log_file), "LR_history.jpg")

    def __call__(self, trainer, *args, **kwargs):
        n_iters = trainer.state.epoch * len(trainer.data_loader)
        plt.plot(np.linspace(1, trainer.state.epoch, n_iters), trainer.lrs_history)
        plt.xlabel("Epochs")
        plt.ylabel("Learning rate")
        plt.savefig(self.filepath, dpi=300)


class GeneralLoggingCallback:
    def __init__(self, logger, validator, print_name: str, patience: int, tolerance: float, min_is_good: bool,
                 metric: str, train_log_file: str):
        """ Includes:
        - Validation on val set
        - Early stopping + Saving of the best model
        """
        self.name = print_name
        self.patience = patience
        self.tolerance = tolerance
        self.validator = validator
        self.silent_epochs = 0
        self.mult_factor = 1 if min_is_good else -1
        self.best_metric = np.inf if min_is_good else -np.inf
        self.es_metric = metric
        self.logger = logger
        self.log_file = logger.log_file
        self.save_dir = os.path.dirname(self.log_file)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(train_log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def __call__(self, trainer, *args, **kwargs):
        metrics_dict = self.validator(trainer.network)

        # log metrics
        msg = f"{self.name} on epoch {trainer.state.epoch}:"
        for metric_name, metric_val in metrics_dict.items():
            msg += f"\t{metric_name} {metric_val:.5f},"
        self.logger.write(msg[:-1])

        # es
        metric_val = metrics_dict[self.es_metric]
        if self.mult_factor * metric_val > self.mult_factor * self.best_metric or \
                np.abs(metric_val - self.best_metric) < self.tolerance:
            self.silent_epochs += 1
            if self.silent_epochs > self.patience:
                logging.info(f"Early stopping at epoch {trainer.state.epoch + 1}")
                trainer.continue_training = False
        else:
            self.best_metric = metric_val
            self.silent_epochs = 0
            torch.save(trainer.network.state_dict(),
                       os.path.join(self.save_dir, f"model_epoch_{trainer.state.epoch}.pth"))
            logging.info(f"Saved best model at epoch {trainer.state.epoch}")

        return metric_val



