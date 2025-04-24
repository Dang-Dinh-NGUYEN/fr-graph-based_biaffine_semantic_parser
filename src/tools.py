import pickle
import sys

import torch
from src.biaffine_parser import biaffine_parser
from torch.utils.data import Dataset


# torch.serialization.add_safe_globals([biaffine_parser])

# --- Customized Arguments Parser ---

def list_of_strings(arg):
    if "None" in arg or len(arg) == 0:
        return None
    return arg.split(',')


def list_of_int(arg):
    return list(map(int, arg.split(',')))


def convert_list_to_string(lst):
    # Convert each element in the list to a string
    return '|'.join(str(x) if x != '' else '_' for x in lst)


# --- Customized Dataset ---
class CustomizedDataset:
    def __init__(self, data, required_keys=None):
        """
        :param data: Dictionary where each key contains a list of samples (column-based structure).
        :param required_keys: A list of expected keys; if None, it uses all keys from the dictionary.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected 'data' to be a dict, got {type(data)}")

        self.data = data
        self.required_keys = required_keys or list(data.keys())
        self.num_samples = len(next(iter(data.values())))  # Get the length from any key

        if any(len(v) != self.num_samples for v in data.values()):
            raise ValueError("All lists in 'data' must have the same length.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns a dictionary containing one sample from each key.
        """
        return {key: self.data[key][idx] for key in self.required_keys}


# --- Customized Batch ---
def dynamic_collate_fn(batch):
    """
    Dynamically collates a batch of variable-length and fixed-size inputs.
    Handles missing values, padding, and tensor stacking.
    """
    batch_keys = batch[0].keys()  # Extract all possible keys
    # print(f"batch keys {batch_keys}")
    collated_batch = {}

    for key in batch_keys:
        values = [sample[key] for sample in batch if sample[key] is not None]
        # print("current value")
        # print(values)

        if len(values) == 0:
            continue  # Skip empty fields

        collated_batch[key] = torch.stack(values)

    return collated_batch


# --- Save/Load models ---

def save_model(file_path: str, model, optimizer, arc_loss_function, label_loss_function, trained_vocabularies, n_epochs, batch_size, history):
    parameters = {
        'model': model,
        'optimizer': optimizer,
        'arc_loss_function': arc_loss_function,
        'label_loss_function': label_loss_function,
        'trained_vocabularies': trained_vocabularies,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'history': history
    }
    with open(file_path, 'wb') as f:
        pickle.dump(parameters, f)
    print(f"Model saved to {file_path}", file=sys.stderr)


def load_model(trained_model_path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(trained_model_path, 'rb') as f:
        parameters = pickle.load(f)

    model = parameters['model']
    model.to(device)  # Move to the specified device

    optimizer = parameters['optimizer']
    arc_loss_function = parameters['arc_loss_function']
    label_loss_function = parameters['label_loss_function']
    trained_vocabularies = parameters['trained_vocabularies']
    n_epochs = parameters['n_epochs']
    batch_size = parameters['batch_size']
    history = parameters['history']

    print(f"Loaded model from {trained_model_path} on {device}", file=sys.stderr)
    return model, optimizer, arc_loss_function, label_loss_function, trained_vocabularies, n_epochs, batch_size, history


# --- Multi-hot Encoding ---
def multi_hot_encode_torch(indices, num_classes):
    """Creates a multi-hot vector from a list of indices (e.g., [0, 1]) or a single int."""
    multi_hot = torch.zeros(num_classes, dtype=torch.long)

    # Make sure indices is always a list
    if isinstance(indices, int):
        indices = [indices]
    elif not isinstance(indices, list):
        indices = list(indices)  # for tuples or tensors

    for idx in indices:
        if 0 <= idx < num_classes:
            multi_hot[int(idx)] = 1

    return multi_hot


# --- Multi-label Encoding ---
def multi_label_encode_torch(indices, num_classes, values):
    """
    Creates a multi-label vector from indices and associated values.
    Each index gets the corresponding value in the output tensor.
    """
    multi_label = torch.zeros(num_classes, dtype=torch.long)

    # Ensure indices and values are lists (or at least iterable)
    if isinstance(indices, int):
        indices = [indices]
    elif not isinstance(indices, list):
        indices = list(indices)

    if isinstance(values, int):
        values = [values]
    elif not isinstance(values, list):
        values = list(values)

    for idx, value in zip(indices, values):
        if 0 <= idx < num_classes:
            multi_label[int(idx)] = value

    return multi_label
# TO DO : PLOT TRAINING CURVES
