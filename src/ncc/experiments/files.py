"""Reading and writing the experiment files: pickles, YAML and CSV."""
import csv
import io
import pickle

import torch
import yaml


class _CPUUnpickler(pickle.Unpickler):
    """Unpickler that loads pickled torch tensors onto the CPU.

    Some base prediction files of the paper hold (unused) tensors saved on a CUDA device, which a
    plain pickle.load cannot restore on a machine without CUDA.
    """

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)


def load_pickle(path):
    with open(path, 'rb') as handle:
        return _CPUUnpickler(handle).load()


def save_pickle(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_yaml(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        csv.writer(f).writerows(rows)
