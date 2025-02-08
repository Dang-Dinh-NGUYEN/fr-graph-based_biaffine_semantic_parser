import os
import sys

import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib import conllulib
import config


def pad_tensor(batchs: list, max_len: int, padding_value: int = config.PAD_TOKEN_VAL, device=None) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = torch.full((len(batchs), max_len), padding_value, dtype=torch.long, device=device)

    for i, sentence in enumerate(batchs):
        result[i, 0] = padding_value  # Add padding ID at the beginning
        result[i, 1:len(sentence) + 1] = torch.tensor(sentence[:], dtype=torch.long, device=device)

    return result


def preprocess_data(
        file_path: str,
        word_vocab: dict = config.WORD_VOCAB,
        tag_vocab: dict = config.TAG_VOCAB,
        update: bool = True,
        max_len: int = 50,
        device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(file_path, "r", encoding="UTF-8") as file:
        tokenLists = conllulib.CoNLLUReader(file).readConllu()

        words, tags, governors = [], [], []

        for tokenList in tqdm(tokenLists, desc="Processed", unit=f" sentence(s)"):
            current_words, current_tags, current_governors = [], [], []

            for token in tokenList:
                if token['form'] not in word_vocab and update:
                    word_vocab[token['form']] = len(word_vocab)
                current_words.append(word_vocab.get(token['form'], word_vocab['UNK_ID']))

                if token['upos'] not in tag_vocab and update:
                    tag_vocab[token['upos']] = len(tag_vocab)
                current_tags.append(tag_vocab.get(token['upos'], tag_vocab['UNK_ID']))

                current_governors.append(token['head'])

            if len(current_words) < max_len:
                words.append(current_words)
                tags.append(current_tags)
                governors.append(current_governors)

        words = pad_tensor(words, max_len, device=device)
        tags = pad_tensor(tags, max_len, device=device)
        governors = pad_tensor(governors, max_len, device=device)

        return word_vocab, tag_vocab, words, tags, governors


def save_preprocessed_data(word_vocab: dict, tag_vocab: dict, words: torch.Tensor, tags: torch.Tensor, governors: torch.Tensor,
                           save_path: str):
    torch.save({
        'word_vocab': word_vocab,
        'tag_vocab': tag_vocab,
        'words': words.cpu(),
        'tags': tags.cpu(),
        'governors': governors.cpu()
    }, save_path)
    print('Saved preprocessed data to {}'.format(save_path))


def load_preprocessed_data(preprocessed_data_path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Loading preprocessed data from {}'.format(preprocessed_data_path))

    preprocessed_data = torch.load(preprocessed_data_path)

    return (
        preprocessed_data['word_vocab'],
        preprocessed_data['tag_vocab'],
        preprocessed_data['words'].to(device),
        preprocessed_data['tags'].to(device),
        preprocessed_data['governors'].to(device)
    )


def display_preprocessed_data(*args):
    print(f"\nDisplaying preprocessed data")
    for value in args:
        print(f"Type: {type(value)}")

        if isinstance(value, dict):
            print(f"Length: {len(value)}")
            if len(value) > 100:
                print(", ".join([f"{key}: {val}" for key, val in list(value.items())[:25]]))
            else:
                print(", ".join([f"{key}: {val}" for key, val in value.items()]))

        elif isinstance(value, list):
            print(f"Length: {len(value)}")
            print(value[:25] if len(value) > 100 else value)

        elif isinstance(value, torch.Tensor):
            print(f"Shape: {value.shape}")
            print(value[:25] if value.shape[0] > 100 else value)

        else:
            print("Length/Shape: Not applicable")

        print()
