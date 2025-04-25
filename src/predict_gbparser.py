import sys
from collections import defaultdict

import conllu
import torch
from tqdm import tqdm
from lib import conllulib
import src.config as cf
from src import tools
from src.modules import TransformerEncoder, RecurrentEncoder


def predict(input_file_path: str, model, trained_vocabularies, semantic_prediction=False,
            display=False, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    with open(input_file_path, 'r', encoding='UTF-8') as f:
        tokenLists = conllulib.CoNLLUReader(f).readConllu(
            field_parsers={"head": lambda line, i: conllu.parser.parse_nullable_value(line[i])})

        for tokenList in tqdm(tokenLists, desc="Processed", unit=" sentence(s)"):
            inputs = prepare_inputs(model, tokenList, trained_vocabularies, device)

            S_arc, S_rel = model(False, *inputs)

            predicted_heads, selected_deprels = process_predictions(S_arc, S_rel, semantic_prediction)

            if display:
                display_predictions(tokenList, predicted_heads, selected_deprels, trained_vocabularies, device)

            # Assign predictions and write to file
            for token, head, label in zip(tokenList, predicted_heads, selected_deprels):
                pred_deprel = get_deprel_from_vocab(trained_vocabularies['deprel_vocab'], label)
                deps = '|'.join([f"{index}:{label}" for index, label in zip(head, pred_deprel)])

                token['deps'] = deps if deps != "" else "_"

            print(tokenList.serialize(), end="")


def prepare_inputs(model, tokenList, trained_vocabularies, device):
    """Prepares model inputs dynamically based on encoder type."""
    if isinstance(model.encoder, TransformerEncoder):
        return prepare_transformer_inputs(model, tokenList, trained_vocabularies, device)
    elif isinstance(model.encoder, RecurrentEncoder):
        return prepare_recurrent_inputs(tokenList, trained_vocabularies, device)
    else:
        raise ValueError(f"Unsupported encoder type: {type(model.encoder)}")


def prepare_transformer_inputs(model, tokenList, trained_vocabularies, device):
    """Handles input processing for Transformer encoders."""
    inputs = {}

    for embedding in model.encoder.embeddings:
        embedding_key = embedding.split('_')[0]  # Extract feature name (e.g., "upos" from "upos_embeddings")
        if f"{embedding_key}_vocab" in trained_vocabularies:
            inputs[embedding_key] = torch.tensor(
                [model.encoder.pad_token_id] + [
                    trained_vocabularies[f"{embedding_key}_vocab"].get(token.get(embedding_key),
                                                                       model.encoder.tokenizer.unk_token_id)
                    for token in tokenList], device=device).unsqueeze(0)

    sentence_forms = [model.encoder.pad_token] + [token['form'] for token in tokenList]
    tokenized_sentence = model.encoder.tokenizer(sentence_forms, return_tensors='pt',
                                                 is_split_into_words=True).to(device)

    subword_ids = tokenized_sentence.word_ids()[1:-1]

    with torch.no_grad():
        subword_embeddings = model.encoder.transformer(**tokenized_sentence)['last_hidden_state'][0]
    subword_embeddings = subword_embeddings[1:-1]  # Remove special tokens

    # Align subwords to original tokens
    aligned_embeddings = defaultdict(list)
    for i, subword_id in enumerate(subword_ids):
        aligned_embeddings[subword_id].append(subword_embeddings[i])

    contextual_embeddings = torch.stack([torch.stack(aligned_embeddings[i]).mean(dim=0)
                                         for i in sorted(aligned_embeddings.keys())]).to(device).unsqueeze(0)

    inputs["contextual_embeddings"] = contextual_embeddings
    return list(inputs.values())


def prepare_recurrent_inputs(tokenList, trained_vocabularies, device):
    """Handles input processing for Recurrent encoders."""
    sentence_forms = [cf.PAD_TOKEN_VAL] + [
        trained_vocabularies['form_vocab'].get(token['form'], cf.UNK_TOKEN_VAL) for token in tokenList
    ]

    upos_tags = [cf.PAD_TOKEN_VAL] + [
        trained_vocabularies['upos_vocab'].get(token['upos'], cf.UNK_TOKEN_VAL) for token in tokenList
    ]

    sentence_forms = torch.tensor(sentence_forms, device=device).unsqueeze(0)
    upos_tags = torch.tensor(upos_tags, device=device).unsqueeze(0)

    return [sentence_forms, upos_tags]


def process_predictions(S_arc, S_rel, semantic_prediction):
    """Processes model predictions: head selection & dependency labels."""
    if semantic_prediction:
        return process_semantic_predictions(S_arc, S_rel)
    return process_syntactic_predictions(S_arc, S_rel)


def process_syntactic_predictions(S_arc, S_rel):
    """Processes model predictions: head selection & dependency labels."""
    S_arc = torch.nn.functional.softmax(S_arc, dim=-1)

    predicted_heads = torch.argmax(S_arc, dim=-1).squeeze(0)[1:]

    # Identify the ROOT position and set it to 0
    # root_position = \
    # (predicted_heads == torch.arange(1, len(predicted_heads) + 1, device=device)).nonzero(as_tuple=True)[0]
    # predicted_heads[root_position] = 0

    # Extract dependency relations
    predicted_deprels = torch.argmax(S_rel, dim=3).squeeze(0)[1:]  # (L, L)
    selected_deprels = predicted_deprels[torch.arange(len(predicted_heads)), predicted_heads]  # (L-1)

    return predicted_heads.tolist(), selected_deprels.tolist()


def process_semantic_predictions(S_arc, S_rel):
    # Step 1: Normalize arc scores (apply sigmoid to get probabilities)
    S_arc = torch.sigmoid(S_arc)  # Convert scores to probabilities

    # Step 2: Multi-head prediction (apply threshold)
    pred_heads = (S_arc > 0.5).to(torch.int).squeeze(0)[1:]  # (L, L)

    head_indices_list = []

    for i in range(pred_heads.shape[0]):
        # Get the indices where pred_heads[i, :] == 1 (head indices)
        head_indices = torch.nonzero(pred_heads[i], as_tuple=True)[0].tolist()
        """
        if not head_indices:  # If no heads are found for the token
            # Find the index with the maximum arc score
            max_head_idx = torch.argmax(S_arc[i, :]).item()
            head_indices = [max_head_idx]  # Assign this as the head
         """
        head_indices_list.append(head_indices)
    predicted_relations = []

    # Iterate over each token
    for i, head_indices in enumerate(head_indices_list):
        token_relations = []

        for head_idx in head_indices:
            # Extract the relation scores between the token i and its head (head_idx)
            # Since S_rel has shape (1, L, L, C), we need to access the score from the correct location
            relation_scores = S_rel[0, i, head_idx]  # Relation scores for token i and head head_idx
            # Find the index of the relation with the highest score
            predicted_relation = torch.argmax(relation_scores).item()
            token_relations.append(predicted_relation)

        predicted_relations.append(token_relations)
    return head_indices_list, predicted_relations


def display_predictions(tokenList, predicted_heads, selected_deprels, trained_vocabularies, device):
    """Displays debug information for predictions."""
    print("Forms:", [token['form'] for token in tokenList])
    print("Predicted Heads:", predicted_heads)

    dependencies = torch.tensor([token['head'] for token in tokenList], device=device).unsqueeze(0)
    print("True Heads:", dependencies.tolist())

    print("Predicted Deprels:", selected_deprels)
    deprel_labels = [trained_vocabularies['deprel_vocab'].get(token['deprel'], cf.UNK_TOKEN_VAL) for token in tokenList]
    print("True Deprels:", deprel_labels)
    print()


def get_deprel_from_vocab(deprel_vocab, label):
    """Retrieves the dependency relation label from the vocabulary."""
    inv_deprel_vocab = {v: k for k, v in deprel_vocab.items()}

    # Convert tensor to list if needed
    if hasattr(label, 'tolist'):
        label = label.tolist()

    return [inv_deprel_vocab[i] for i in label]
