import io
import pickle

import torch
from src.biaffine_parser import biaffine_parser  # Ensure the class is imported

torch.serialization.add_safe_globals([biaffine_parser])


def save_model(file_path: str, model, optimizer, criterion, trained_words, trained_tags, trained_labels, n_epochs, batch_size,
               history):
    parameters = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'trained_words': trained_words,
        'trained_tags': trained_tags,
        'trained_labels': trained_labels,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'history': history
    }
    with open(file_path, 'wb') as f:
        pickle.dump(parameters, f)
    print(f"Model saved to {file_path}")


def load_model(trained_model_path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(trained_model_path, 'rb') as f:
        parameters = pickle.load(f)  # Ensures CPU compatibility

    model = parameters['model']
    model.to(device)  # Move to the specified device

    optimizer = parameters['optimizer']
    criterion = parameters['criterion']
    trained_words = parameters['trained_words']
    trained_tags = parameters['trained_tags']
    trained_labels = parameters['trained_labels']

    print(f"Loaded model from {trained_model_path} on {device}")
    return model, optimizer, criterion, trained_words, trained_tags, trained_labels
