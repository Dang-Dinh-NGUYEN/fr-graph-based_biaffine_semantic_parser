import pprint
from collections import defaultdict

import torch
from tqdm import tqdm
import src.config as cf
import lib.conllulib
import conllu
import src.tools as tools


# --- Tools ---
def pad_tensor(seq: list, max_len: int, pad_value: int = cf.PAD_TOKEN_VAL, device=None) -> torch.Tensor:
    """
    Pad sequences by prepending a designated padding value, and then append additional padding values until each
    sequence reaches the maximum length.
    :param seq: the input sequence to be padded
    :param max_len: the maximum length of the padded sequence
    :param pad_value: pad value to be prepended/added
    :param device: device on which the tensor will be stored
    :return: a torch.Tensor of padded sequence
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    padded_seq = torch.full((len(seq), max_len), pad_value, dtype=torch.long, device=device)

    for i, sent in enumerate(seq):
        padded_seq[i, 0] = pad_value  # Add padding ID at the beginning of the sentence
        padded_seq[i, 1:len(sent) + 1] = sent.clone().detach().to(device)

    return padded_seq


def pad_and_stack_2d(seq: list[torch.Tensor], max_len: int, pad_value: float = 0.0) -> torch.Tensor:
    """
    Pads a list of 1D tensors (each of length max_len) into a stacked 2D tensor of shape (max_len, max_len).
    Prepends a zero tensor, and appends padding until the desired number of rows is reached.
    """
    if not seq:
        return torch.zeros((max_len, max_len), dtype=torch.long)

    d = seq[0].shape[0]  # Width of each 1D tensor
    zero = torch.full((d,), pad_value, dtype=seq[0].dtype)

    padded = [zero] + seq
    padded = padded[:max_len]  # truncate if too long
    padded += [zero] * (max_len - len(padded))  # pad if too short

    return torch.stack(padded)

def save_preprocessed_data(data, file_path):
    torch.save(data, file_path)
    print(f"Saved preprocessed data to {file_path}")


def load_preprocessed_data(preprocessed_data_path: str):
    print(f"Loading preprocessed data from {preprocessed_data_path}")
    return torch.load(preprocessed_data_path)


# --- Preprocess data ---
def preprocess_data(file_path: str, columns: list, vocabularies=None,
                    tokenizer=None, pre_trained_model=None,
                    update=True,
                    pad_value: int = cf.PAD_TOKEN_VAL,
                    unk_value: int = cf.UNK_TOKEN_VAL,
                    max_len: int = 50,
                    device=None):
    if vocabularies is None:
        vocabularies = {}

    if columns is None:
        columns = []
    else:
        assert all(col in cf.UD_COLUMNS for col in columns), "Invalid column(s) in input"
        if 'deps' in columns:
            columns += [col for col in ['head', 'deprel'] if col not in columns]

    for col in columns:
        if col not in ['id', 'head', 'deps']:
            vocabularies.setdefault(f"{col}_vocab", {"PAD_ID": pad_value, "UNK_ID": unk_value})

    with open(file_path, "r", encoding="UTF-8") as file:
        tokenLists = lib.conllulib.CoNLLUReader(file).readConllu(
            field_parsers={"head": lambda line, i: conllu.parser.parse_nullable_value(line[i])}
        )

        extracted_values = {f"extracted_{col}": [] for col in columns if col != 'deps'}
        if tokenizer and pre_trained_model:
            extracted_values["extracted_contextual_embeddings"] = []

        for tokenList in tqdm(tokenLists, desc="Processed", unit=" sentence(s)"):
            current_values = {f"current_{col}": [] for col in columns if col != 'deps'}
            current_sentence = [token['form'] for token in tokenList]
            for token in tokenList:
                for col in columns:
                    if col == 'deps':
                        for rel, head in token['deps']:
                            if update and rel not in vocabularies["deprel_vocab"]:
                                vocabularies["deprel_vocab"][rel] = len(vocabularies["deprel_vocab"])
                            current_values['current_deprel'].append(
                                vocabularies["deprel_vocab"].get(rel, unk_value)
                            )
                            current_values['current_head'].append(head)
                    elif col in ['head', 'deprel']:
                        continue
                    else:
                        col_name = f"{col}_vocab"
                        if update and token[col] not in vocabularies[col_name]:
                            vocabularies[col_name][token[col]] = len(vocabularies[col_name])
                        current_values[f"current_{col}"].append(
                            vocabularies[col_name].get(token[col], unk_value)
                        )

            if len(current_sentence) < max_len:
                for col in columns:
                    if col == 'deps':
                        continue
                    elif col == 'head':
                        encoded_heads = [tools.multi_hot_encode_torch(h, max_len) for h in
                                         current_values['current_head']]
                        extracted_values["extracted_head"].append(pad_and_stack_2d(encoded_heads, max_len))
                    elif col == 'deprel':
                        encoded_deprel = [
                            tools.multi_label_encode_torch(h, max_len, rels)
                            for rels, h in zip(current_values['current_deprel'], current_values['current_head'])
                        ]
                        extracted_values["extracted_deprel"].append(pad_and_stack_2d(encoded_deprel, max_len))
                    else:
                        extracted_values[f"extracted_{col}"].append(
                            torch.tensor(current_values[f"current_{col}"], dtype=torch.long)
                        )

                if tokenizer and pre_trained_model:
                    embeddings = compute_contextual_embeddings(
                        sentence=current_sentence,
                        tokenizer=tokenizer,
                        pretrained_model=pre_trained_model,
                        max_len=max_len,
                        device=device
                    )
                    extracted_values["extracted_contextual_embeddings"].append(embeddings)

        # Final padding
        for key in extracted_values:
            if key in ["extracted_head", "extracted_deprel"]:
                extracted_values[key] = torch.stack(extracted_values[key])
            elif key == "extracted_contextual_embeddings":
                extracted_values[key] = torch.stack(extracted_values[key]).to("cpu")
            else:
                extracted_values[key] = pad_tensor(extracted_values[key], pad_value=pad_value,
                                                   max_len=max_len).to("cpu")

    return {**extracted_values, **vocabularies}


def compute_contextual_embeddings(sentence: list, tokenizer, pretrained_model, max_len: int = 50,
                                  device=None) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentence = [tokenizer.pad_token] + sentence + [tokenizer.pad_token] * (
            max_len - len(sentence) - 1)  # Pad input the input sentence

    tokenized_sentence = tokenizer(sentence, return_tensors="pt", is_split_into_words=True).to(device)
    subword_ids = tokenized_sentence.word_ids()
    subword_ids = subword_ids[1:-1]  # Remove BOS and EOS tokens

    with torch.no_grad():
        subword_embeddings = pretrained_model(**tokenized_sentence)['last_hidden_state'][0]

    subword_embeddings = subword_embeddings[1:-1]  # Remove BOS and EOS tokens

    # Align sub-words embeddings
    aligned_subword_emb = defaultdict(list)
    for i, subword in enumerate(subword_ids):
        aligned_subword_emb[subword].append(subword_embeddings[i])

    # Compute contextual embedding for each aligned subword
    contextual_embedding = torch.stack([torch.stack(aligned_subword_emb[i]).mean(dim=0)
                                        for i in sorted(aligned_subword_emb.keys())]).to("cpu")

    return contextual_embedding


# --- Display ---
def display_preprocessed_data(data, max_rows=25):
    pp = pprint.PrettyPrinter()

    for key, value in data.items():
        print(f"\n ---{key}---")

        if isinstance(value, torch.Tensor):
            print("(Tensor shape:", value.shape, ")")
            print(torch.tensor(value.tolist()[:max_rows]))  # Convert tensor to list for easy printing

        elif isinstance(value, dict):
            print("(Dict size:", len(value), ")")
            sample_items = list(value.items())[:max_rows]
            pp.pprint(dict(sample_items))  # Pretty-print dictionary

        elif isinstance(value, list):  # Handling contextual embeddings (if stored as lists)
            print(value[:max_rows])

        else:
            print("(Unknown Type)")
            print(value)
