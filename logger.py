import os
import logging


def save_options(args, filepath):
    with open(filepath, 'w') as f:
        for arg, value in sorted(vars(args).items()):
            f.write("%s:\t%r\n" % (arg, value))


class Logger:
    def __init__(self, log_file=None):
        if log_file is None:
            self.log_file = None
            logging.info("Logs will appear in the console and won't be saved")
        else:
            self.log_file = log_file
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            logging.info(f"Logs will appear in the {log_file}.")

    def write(self, msg):
        if self.log_file is None: print(msg)
        else:
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')
