import csv
import pickle
import numpy as np
import yaml


def pickle_save(fname, data):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(fname):
    with open(fname, 'rb') as handle:
        return pickle.load(handle)


def load_yaml_params(params_file_path):
    with open(params_file_path, 'r') as stream:
        return yaml.safe_load(stream)


def WriteRows2CSV(rows: list, output_path):
    with open(output_path, 'w+') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(rows)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given
    patience.
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Early stopping to stop the training when the loss does not improve after
        certain epochs.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.model_state_dict = None

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decreases.
        """
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        self.model_state_dict = model.state_dict()
        self.val_loss_min = val_loss
