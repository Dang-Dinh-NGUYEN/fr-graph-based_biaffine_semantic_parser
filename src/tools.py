import pickle
import torch
from src.biaffine_parser import biaffine_parser

torch.serialization.add_safe_globals([biaffine_parser])


def list_of_strings(arg):
    return arg.split(',')


def list_of_int(arg):
    return list(map(int, arg.split(',')))


def save_model(file_path: str, model, optimizer, criterion, trained_vocabularies, n_epochs, batch_size, history):
    parameters = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'trained_vocabularies': trained_vocabularies,
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
        parameters = pickle.load(f)

    model = parameters['model']
    model.to(device)  # Move to the specified device

    optimizer = parameters['optimizer']
    criterion = parameters['criterion']
    trained_vocabularies = parameters['trained_vocabularies']
    n_epochs = parameters['n_epochs']
    batch_size = parameters['batch_size']
    history = parameters['history']

    print(f"Loaded model from {trained_model_path} on {device}")
    return model, optimizer, criterion, trained_vocabularies, n_epochs, batch_size, history

# TO DO : PLOT TRAINING CURVES
