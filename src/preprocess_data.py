import pprint
from collections import defaultdict

import torch
from tqdm import tqdm
import src.config as cf
import lib.conllulib


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

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    padded_seq = torch.full((len(seq), max_len), pad_value, dtype=torch.long, device=device)

    for i, sent in enumerate(seq):
        padded_seq[i, 0] = pad_value  # Add padding ID at the beginning of the sentence
        padded_seq[i, 1:len(sent) + 1] = sent.clone().detach().to(device)

    return padded_seq


def preprocess_data(file_path: str, columns: list, vocabularies=None,         
                    tokenizer=None, pre_trained_model=None, 
                    update=True, 
                    pad_value: int = cf.PAD_TOKEN_VAL,
                    unk_value: int = cf.UNK_TOKEN_VAL, 
                    max_len: int = 50, 
                    device=None):
    """
    Preprocess data from CoNLL-U files.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if vocabularies is None:
        vocabularies = {}
    
    if columns is None:
        columns = []
    else:
        assert all(col in cf.UD_COLUMNS for col in columns), "Invalid column(s) in input"

    for col in columns:
        if col in ['head']:
            continue
        elif vocabularies.get(f"{col.split(':')[-1]}_vocab") is None:
            vocabularies[f"{col.split(':')[-1]}_vocab"] = {"PAD_ID": pad_value, "UNK_ID": unk_value}

    with open(file_path, "r", encoding="UTF-8") as file:
        tokenLists = lib.conllulib.CoNLLUReader(file).readConllu()

        extracted_values = {f"extracted_{col}": [] for col in columns}
        if tokenizer and pre_trained_model:
            extracted_values["extracted_contextual_embeddings"] = []

        for tokenList in tqdm(tokenLists, desc="Processed", unit=" sentence(s)"):
            current_values = {f"current_{col}": [] for col in columns if col != 'id'}

            current_sentence = [token['form'] for token in tokenList]

            for token in tokenList:
                for col in columns:
                    col_name = f"{col.split(':')[-1]}_vocab"

                    if col == 'head':
                        current_values[f"current_{col}"].append(token[col])
                    else:
                        if col_name not in vocabularies:
                            vocabularies[col_name] = {"PAD_ID": pad_value, "UNK_ID": unk_value}

                        if token[col] not in vocabularies[col_name] and update:
                            vocabularies[col_name][token[col]] = len(vocabularies[col_name])

                        current_values[f"current_{col}"].append(
                            vocabularies[col_name].get(token[col], vocabularies[col_name]["UNK_ID"])
                        )

            if len(current_sentence) < max_len:
                for col in columns:
                    extracted_values[f"extracted_{col}"].append(
                        torch.tensor(current_values[f"current_{col}"], dtype=torch.long)
                    )
                if tokenizer and pre_trained_model:
                    current_contextual_embeddings = compute_contextual_embeddings(
                        sentence=current_sentence, tokenizer=tokenizer,
                        pretrained_model=pre_trained_model, max_len=max_len, device=device
                    )

                    extracted_values["extracted_contextual_embeddings"].append(current_contextual_embeddings)

        for extracted in extracted_values.keys():
            if extracted == "extracted_contextual_embeddings":
                extracted_values[extracted] = torch.stack(extracted_values[extracted]).to("cpu")
            else:
                extracted_values[extracted] = pad_tensor(extracted_values[extracted], pad_value=pad_value,
                                                         max_len=max_len).to("cpu")

    preprocessed_data = {**extracted_values, **vocabularies}
    return preprocessed_data


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


def save_preprocessed_data(data, file_path):
    torch.save(data, file_path)
    print(f"Saved preprocessed data to {file_path}")


def load_preprocessed_data(preprocessed_data_path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading preprocessed data from {preprocessed_data_path}")

    preprocessed_data = torch.load(preprocessed_data_path)

    return preprocessed_data


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
