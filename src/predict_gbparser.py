import torch
from tqdm import tqdm

from lib import conllulib
import src.config as cf


def predict(input_file_path: str, output_file_path: str, model, trained_forms, trained_upos, trained_deprels,
            display=False, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    with open(input_file_path, 'r', encoding='UTF-8') as f:
        tokenLists = conllulib.CoNLLUReader(f).readConllu()

        with open(output_file_path, 'w', encoding='UTF-8') as output_file:
            for tokenList in tqdm(tokenLists, desc="Processed", unit=f" sentence(s)"):
                # Add PAD_TOKEN at the beginning of the sentence
                sentence_forms = [cf.PAD_TOKEN_VAL] + [trained_forms.get(token['form'], cf.UNK_TOKEN_VAL) for
                                                       token in tokenList]
                upos_tags = [cf.PAD_TOKEN_VAL] + [trained_upos.get(token['upos'], cf.UNK_TOKEN_VAL) for token in
                                                  tokenList]

                # Convert to torch.Tensor
                sentence_forms = torch.tensor(sentence_forms, device=device).unsqueeze(0)
                upos_tags = torch.tensor(upos_tags, device=device).unsqueeze(0)

                S_arc, deprels = model(sentence_forms, upos_tags)

                # Eliminate the first token to obtain the correct predictions
                predicted_heads = torch.argmax(S_arc, dim=2).squeeze(0)[1:]

                # Find the correct ROOT position
                root_position = \
                    (predicted_heads == torch.arange(1, len(predicted_heads) + 1, device=device)).nonzero(
                        as_tuple=True)[0]

                # Set that position to 0 (ROOT)
                predicted_heads[root_position] = 0

                # Step 4: Predict deprels
                predicted_deprels = torch.argmax(deprels, dim=3).squeeze(0)[1:]  # (L, L)

                # Step 5: Select deprels for predicted head-word pairs
                selected_deprels = predicted_deprels[torch.arange(len(predicted_heads)), predicted_heads]  # (L-1)

                if display:
                    print([token['form'] for token in tokenList])

                    print(predicted_heads)
                    dependencies = torch.tensor([token['head'] for token in tokenList], device=device).unsqueeze(0)
                    print(dependencies)

                    print(selected_deprels)
                    deprels = torch.tensor(
                        [trained_deprels.get(token['deprel'], cf.UNK_TOKEN_VAL) for token in tokenList],
                        device=device).unsqueeze(0)
                    print(deprels)
                    print()

                predicted_heads = predicted_heads.tolist()
                selected_deprels = selected_deprels.tolist()

                for token, head, label in zip(tokenList, predicted_heads, selected_deprels):
                    token['head'] = head
                    token['deprel'] = list(trained_deprels.keys())[list(trained_deprels.values()).index(label)]

                # Write the serialized token list to the output file
                output_file.write(tokenList.serialize())
