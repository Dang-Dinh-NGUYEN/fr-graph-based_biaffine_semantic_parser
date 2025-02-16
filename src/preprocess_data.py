import torch
from tqdm import tqdm
import src.config as cf
import lib.conllulib


def pad_tensor(batchs: list, max_len: int, padding_value: int = cf.PAD_TOKEN_VAL, device=None) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = torch.full((len(batchs), max_len), padding_value, dtype=torch.long, device=device)

    for i, sentence in enumerate(batchs):
        result[i, 0] = padding_value  # Add padding ID at the beginning
        result[i, 1:len(sentence) + 1] = torch.tensor(sentence[:], dtype=torch.long, device=device)

    return result


def preprocess_data(
        file_path: str,
        form_vocab: dict = cf.FORM_VOCAB,
        upos_vocab: dict = cf.UPOS_VOCAB,
        deprel_vocab: dict = cf.DEPREL_VOCAB,
        update: bool = True,
        max_len: int = 50,
        device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(file_path, "r", encoding="UTF-8") as file:
        tokenLists = lib.conllulib.CoNLLUReader(file).readConllu()

        forms, upos, heads, deprels = [], [], [], []

        for tokenList in tqdm(tokenLists, desc="Processed", unit=f" sentence(s)"):
            current_forms, current_upos, current_heads, current_deprel = [], [], [], []

            for token in tokenList:
                if token['form'] not in form_vocab and update:
                    form_vocab[token['form']] = len(form_vocab)
                current_forms.append(form_vocab.get(token['form'], form_vocab['UNK_ID']))

                if token['upos'] not in upos_vocab and update:
                    upos_vocab[token['upos']] = len(upos_vocab)
                current_upos.append(upos_vocab.get(token['upos'], upos_vocab['UNK_ID']))

                current_heads.append(token['head'])

                if token['deprel'] not in deprel_vocab and update:
                    deprel_vocab[token['deprel']] = len(deprel_vocab)
                current_deprel.append(deprel_vocab.get(token['deprel'], deprel_vocab['UNK_ID']))

            if len(current_forms) < max_len:
                forms.append(current_forms)
                upos.append(current_upos)
                heads.append(current_heads)
                deprels.append(current_deprel)

        forms = pad_tensor(forms, max_len, device=device)
        upos = pad_tensor(upos, max_len, device=device)
        heads = pad_tensor(heads, max_len, device=device)
        deprels = pad_tensor(deprels, max_len, device=device)

        return form_vocab, upos_vocab, deprel_vocab, forms, upos, heads, deprels


def save_preprocessed_data(form_vocab: dict, upos_vocab: dict, deprel_vocab: dict,
                           forms: torch.Tensor, upos: torch.Tensor, heads: torch.Tensor, deprels: torch.Tensor,
                           save_path: str):
    torch.save({
        'form_vocab': form_vocab,
        'upos_vocab': upos_vocab,
        'deprel_vocab': deprel_vocab,
        'forms': forms.cpu(),
        'upos': upos.cpu(),
        'heads': heads.cpu(),
        'deprels': deprels.cpu()
    }, save_path)
    print('Saved preprocessed data to {}'.format(save_path))


def load_preprocessed_data(preprocessed_data_path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Loading preprocessed data from {}'.format(preprocessed_data_path))

    preprocessed_data = torch.load(preprocessed_data_path)

    return (
        preprocessed_data['form_vocab'],
        preprocessed_data['upos_vocab'],
        preprocessed_data['deprel_vocab'],
        preprocessed_data['forms'].to(device),
        preprocessed_data['upos'].to(device),
        preprocessed_data['heads'].to(device),
        preprocessed_data['deprels'].to(device)
    )


def display_preprocessed_data(*args):
    print(f"\nDisplaying preprocessed data")
    for value in args:
        print(f"Type: {type(value)}")

        if isinstance(value, dict):
            print(f"Length: {len(value)}")
            items = list(value.items())[:25] if len(value) > 100 else value.items()
            print(", ".join(f"{k}: {v}" for k, v in items))

        elif isinstance(value, list):
            print(f"Length: {len(value)}")
            print(value[:25] if len(value) > 100 else value)

        elif isinstance(value, torch.Tensor):
            print(f"Shape: {value.shape}")
            print(value[:25] if value.shape[0] > 100 else value)

        else:
            print("Length/Shape: Not applicable")

        print()
