import io
import os
import pickle
import sys

import torch
from torch.autograd import Variable
from tqdm import tqdm

from lib import conllulib
import src.config as cf


def predict(input_file_path: str, output_file_path: str, model, trained_words, trained_tags, trained_label, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    with open(input_file_path, 'r', encoding='UTF-8') as f:
        tokenLists = conllulib.CoNLLUReader(f).readConllu()

        with open(output_file_path, 'w', encoding='UTF-8') as output_file:
            for tokenList in tqdm(tokenLists, desc="Processed", unit=f" sentence(s)"):
                # Add PAD_TOKEN at the beginning of the sentence
                sentence_forms = [cf.PAD_TOKEN_VAL] + [trained_words.get(token['form'], cf.UNK_TOKEN_VAL) for
                                                       token in tokenList]
                upos_tags = [cf.PAD_TOKEN_VAL] + [trained_tags.get(token['upos'], cf.UNK_TOKEN_VAL) for token in
                                                  tokenList]
                dependencies = torch.tensor([token['head'] for token in tokenList], device=device).unsqueeze(0)
                deprels = torch.tensor([trained_label.get(token['deprel'], cf.UNK_TOKEN_VAL) for token in tokenList], device=device).unsqueeze(0)

                # Convert to torch.Tensor
                sentence_forms = torch.tensor(sentence_forms, device=device).unsqueeze(0)
                upos_tags = torch.tensor(upos_tags, device=device).unsqueeze(0)

                S_arc, labels = model(sentence_forms, upos_tags)
                print(S_arc.shape, labels.shape)
                # Eliminate the first token to obtain the correct predictions
                predicted_heads = torch.argmax(S_arc, dim=2).squeeze(0)[1:]
                print(f"Predicted heads shape: {predicted_heads.shape}")

                # Find the correct ROOT position
                root_position = \
                    (predicted_heads == torch.arange(1, len(predicted_heads) + 1, device=device)).nonzero(
                        as_tuple=True)[0]

                # Set that position to 0 (ROOT)
                predicted_heads[root_position] = 0

                # Step 4: Predict labels
                predicted_labels = torch.argmax(labels, dim=3).squeeze(0)  # Shape: [L, L]
                print(f"Predicted labels shape: {predicted_labels.shape}")
                print(predicted_labels)

                # Step 5: Select labels for predicted head-word pairs
                selected_labels = predicted_labels[torch.arange(len(predicted_heads)), predicted_heads]  # Shape: [L-1]
                print(f"Final predicted labels shape: {selected_labels.shape}")

                print([token['form'] for token in tokenList])
                print(predicted_heads)
                print(dependencies)
                print(selected_labels)
                print(deprels)
                print([token['deprel']for token in tokenList])
                print(trained_label)
                print()

                predicted_heads = predicted_heads.tolist()
                selected_labels = selected_labels.tolist()

                for token, head, label in zip(tokenList, predicted_heads, selected_labels):
                    token['head'] = head
                    adjusted_label = 2 if label in [0, 1] else label
                    token['deprel'] = list(trained_label.keys())[list(trained_label.values()).index(adjusted_label)]

                # Write the serialized token list to the output file
                output_file.write(tokenList.serialize())
