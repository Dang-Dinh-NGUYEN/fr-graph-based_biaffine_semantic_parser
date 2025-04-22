import argparse
import os
import sys
import time
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

from src.modules import TransformerEncoder, RecurrentEncoder, Embedding
from src.tools import list_of_strings, list_of_int
from src import preprocess_data, biaffine_parser, predict_gbparser, train_gbparser, tools
from src import config as cf


def handle_preprocess(args):
    """Handles data preprocessing mode"""
    print(f"Preprocessing file : {args.input_file}", file=sys.stderr)

    tokenizer = pre_trained_model = None
    pad_value = cf.PAD_TOKEN_VAL
    unk_value = cf.UNK_TOKEN_VAL

    if args.transformer:
        tokenizer = AutoTokenizer.from_pretrained(args.transformer)
        pre_trained_model = AutoModel.from_pretrained(args.transformer)
        pre_trained_model.to(device)

        pad_value = tokenizer.pad_token_id
        unk_value = tokenizer.unk_token_id

    if args.load is None:
        preprocessed_data = \
            preprocess_data.preprocess_data(file_path=args.input_file, columns=args.columns,
                                            tokenizer=tokenizer, pre_trained_model=pre_trained_model,
                                            pad_value=pad_value,
                                            unk_value=unk_value,
                                            max_len=args.max_len, update=args.update)

    else:
        data = preprocess_data.load_preprocessed_data(args.load)
        vocabularies = {key: value for key, value in data.items() if key.endswith("_vocab")}
        preprocessed_data = \
            preprocess_data.preprocess_data(file_path=args.input_file, columns=args.columns, vocabularies=vocabularies,
                                            tokenizer=tokenizer, pre_trained_model=pre_trained_model,
                                            pad_value=pad_value,
                                            unk_value=unk_value,
                                            max_len=args.max_len, update=args.update)

    if args.save:
        directory, filename = os.path.split(args.input_file)
        output_file = f"preprocessed_{filename}.pt"
        output_file = os.path.join(directory, output_file)
        preprocess_data.save_preprocessed_data(preprocessed_data, output_file)

    if args.display:
        preprocess_data.display_preprocessed_data(preprocessed_data)


def handle_train(args):
    """Handles training mode"""
    tokenizer = pre_trained_model = None
    pad_value = cf.PAD_TOKEN_VAL
    unk_value = cf.UNK_TOKEN_VAL
    if args.encoder_type == "almanach/camembert-base":
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_type)
        pre_trained_model = AutoModel.from_pretrained(args.encoder_type)
        pre_trained_model = pre_trained_model.to(device)

        pad_value = tokenizer.pad_token_id
        unk_value = tokenizer.unk_token_id

    if args.ltrain:
        train_data = preprocess_data.load_preprocessed_data(args.ltrain)
    else:
        train_data = preprocess_data.preprocess_data(file_path=args.ftrain,
                                                     tokenizer=tokenizer, pre_trained_model=pre_trained_model,
                                                     columns=['deps'] + args.embeddings,
                                                     pad_value=pad_value,
                                                     unk_value=unk_value,
                                                     max_len=50, update=True)

    vocabularies = {key: value for key, value in train_data.items() if key.endswith("_vocab")}

    if args.ldev:
        dev_data = preprocess_data.load_preprocessed_data(args.ldev)
    else:
        dev_data = preprocess_data.preprocess_data(file_path=args.fdev, vocabularies=vocabularies,
                                                   tokenizer=tokenizer, pre_trained_model=pre_trained_model,
                                                   columns=['deps'] + args.embeddings,
                                                   pad_value=pad_value,
                                                   unk_value=unk_value,
                                                   max_len=50, update=False)

    deprel_vocab_train = train_data['deprel_vocab']

    embeddings = {}
    if args.embeddings and args.embeddings_dim:
        assert len(args.embeddings) == len(args.embeddings_dim)
        for emb, emb_dim in zip(args.embeddings, args.embeddings_dim):
            num_emb = len(train_data[f"{emb}_vocab"])
            embeddings[f"{emb}_embeddings"] = Embedding(num_emb=num_emb, emb_dim=emb_dim)

    if pre_trained_model and tokenizer:
        encoder = TransformerEncoder(tokenizer=tokenizer, pre_trained_model=pre_trained_model, embeddings=embeddings,
                                     unfreeze=args.unfreeze)
    else:
        encoder = RecurrentEncoder(embeddings=embeddings, hidden_size=args.d_h,
                                   rnn_type=args.encoder_type, num_layers=args.rnn_layers,
                                   bidirectional=args.bidirectional, dropout=args.dropout_rate)

    DependencyParser = biaffine_parser.biaffine_parser(encoder=encoder, embeddings=embeddings,
                                                       num_deprels=len(deprel_vocab_train),
                                                       d_arc=args.d_arc, d_rel=args.d_rel,
                                                       dropout_rate=args.dropout_rate)
    print(DependencyParser, file=sys.stderr)

    optimizer = torch.optim.Adam(DependencyParser.parameters(), lr=args.lr)

    if args.semantic:
        arc_loss_function = nn.BCEWithLogitsLoss()
    else:
        arc_loss_function = nn.CrossEntropyLoss()
    label_loss_function = nn.CrossEntropyLoss()

    trainer = train_gbparser.Trainer(DependencyParser, optimizer, arc_loss_function, label_loss_function, args.batch_size, args.patience)

    start_time = time.time()
    history = trainer.train(train_data, dev_data, args.n_epochs)
    end_time = time.time()
    print(f"Training time {end_time - start_time}", file=sys.stderr)

    if args.save:
        tools.save_model(args.save, model=DependencyParser, optimizer=optimizer,
                         arc_loss_function=arc_loss_function, label_loss_function=label_loss_function,
                         trained_vocabularies=vocabularies,
                         n_epochs=args.n_epochs, batch_size=args.batch_size, history=history)


def handle_predict(args):
    print(f"Loading model from {args.model}", file=sys.stderr)
    model, optimizer, arc_loss_function, label_loss_function, trained_vocabularies, n_epochs, batch_size, history = \
        tools.load_model(trained_model_path=args.model, device=device)

    predict_gbparser.predict(model=model, input_file_path=args.input_file,
                             trained_vocabularies=trained_vocabularies, semantic_prediction=args.semantic,
                             display=args.display, device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Graph-based biaffine parser of French")
    parser.add_argument('--disable_cuda', action='store_true', help='disable CUDA')
    parser.add_argument('--semantic', action='store_true', help='enable semantic parser (default = syntactic')

    mode_parsers = parser.add_subparsers(dest="mode", required=True, help="Select mode: preprocess | train | predict")

    # --- Preprocess Mode ---
    preprocess_parser = mode_parsers.add_parser("preprocess", help="Preprocess data")
    preprocess_parser.add_argument("input_file", type=str, help="path to input file")
    preprocess_parser.add_argument('--columns', '-c', type=list_of_strings, default=['form', 'upos', 'deps'],
                                   help='column.s to be extracted (default = form, upos, head, deprel)')
    preprocess_parser.add_argument('--update', '-u', action='store_true', help='update vocabulary during preprocessing')
    preprocess_parser.add_argument('--transformer', choices=[None, 'almanach/camembert-base'], default=None,
                                   help='name of pre_trained transformer to be used (default=None)')
    preprocess_parser.add_argument('--max_len', '-m', type=int, default=50, help='maximum sequence length (default=50)')
    preprocess_parser.add_argument('--save', '-s', action='store_true', help='save preprocessed data')
    preprocess_parser.add_argument('--load', '-l', type=str, help='path to pre-processed file')
    preprocess_parser.add_argument('--display', '-d', action='store_true', help='display preprocessed data')

    preprocess_parser.set_defaults(func=handle_preprocess)

    # --- Train Mode ---
    train_parser = mode_parsers.add_parser("train", help="Train model")

    train_parser.add_argument('--ftrain', type=str, default=cf.SEQUOIA_SIMPLE_TRAIN,
                              help="path to train corpus")
    train_parser.add_argument('--fdev', type=str, default=cf.SEQUOIA_SIMPLE_DEV, help="path to dev corpus")
    train_parser.add_argument('--ltrain', type=str, help="path to preprocessed train file")
    train_parser.add_argument('--ldev', type=str, help="path to preprocessed dev file")

    train_parser.add_argument('--d_arc', type=int, default=400, help="dimension of head/dependent vector (default=400)")
    train_parser.add_argument('--d_rel', type=int, default=100, help="dimension of deprel vector (default=100)")

    encoder_subparser = train_parser.add_subparsers(dest="encoder_type", required=True,
                                                    help="encoder type: lstm | gru | almanach/camembert-base")

    # --- RNN parser ---
    rnn_parser = argparse.ArgumentParser(add_help=False)
    rnn_parser.add_argument('--embeddings', '-e', type=list_of_strings, default=["form", "upos"],
                            help='supplementary embeddings (default = form, upos)')
    rnn_parser.add_argument('--embeddings_dim', '-ed', type=list_of_int, default=[100, 100],
                            help='dimensions of supplementary embeddings (default = 100, 100)')
    rnn_parser.add_argument('--d_h', type=int, default=200, help="dimension of recurrent state (default=200)")
    rnn_parser.add_argument('--rnn_layers', type=int, default=3, help="number of rnn's layer (default=3)")
    rnn_parser.add_argument('--bidirectional', action='store_true', help='enable bidirectional')

    # --- LSTM parser ---
    lstm_parser = encoder_subparser.add_parser('lstm', help="lstm encoder", parents=[rnn_parser])

    # --- GRU parser ---
    gru_parser = encoder_subparser.add_parser('gru', help="gru encoder", parents=[rnn_parser])

    # --- CamemBERT parser ---
    camembert_parser = encoder_subparser.add_parser('almanach/camembert-base', help="camembert encoder")
    camembert_parser.add_argument('--embeddings', '-e', type=list_of_strings, default=[],
                                  help='supplementary embeddings')
    camembert_parser.add_argument('--embeddings_dim', '-ed', type=list_of_int, default=[],
                                  help='dimensions of supplementary embeddings')
    camembert_parser.add_argument('--unfreeze', type=int, default=0, help="last n layers to be fine tuned (default=0)")

    train_parser.add_argument('--dropout_rate', '-r', type=float, default=0.33, help="dropout rate (default=0.33)")
    train_parser.add_argument('--n_epochs', type=int, default=30, help="number of training epochs")
    train_parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    train_parser.add_argument('--lr', type=float, default=0.002, help="learning rate")
    train_parser.add_argument("--patience", "-p", type=int, default=5, help="number of patiences")
    train_parser.add_argument('--save', '-s', type=str, default=None, help="path to save model")

    train_parser.set_defaults(func=handle_train)

    # --- Predict Mode ---
    predict_parser = mode_parsers.add_parser("predict", help="Predict semantic structures")

    predict_parser.add_argument('input_file', type=str, help="path to input file")
    # predict_parser.add_argument('output_file', type=str, help="path to output file")
    predict_parser.add_argument('model', type=str, help="path to trained model")
    predict_parser.add_argument('--display', '-d', action='store_true', help='display the predictions')

    predict_parser.set_defaults(func=handle_predict)

    args = parser.parse_args()

    if args.disable_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f"Using {device}", file=sys.stderr)

    args.func(args)
