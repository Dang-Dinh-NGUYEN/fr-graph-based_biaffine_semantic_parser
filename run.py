import argparse
import os

import torch
from torch import nn

from src import preprocess_data, biaffine_parser, predict_gbparser, train_gbparser, tools
from src import config as cf


def handle_preprocess(args):
    """Handles data preprocessing mode"""
    print(f"Preprocessing file : {args.input_file}")

    if args.load is None:
        form_vocab, upos_vocab, deprel_vocab, forms, upos, heads, deprels = \
                    preprocess_data.preprocess_data(file_path=args.input_file, update=args.update, max_len=args.max_len)
    else:
        preprocessed_form_vocab, preprocessed_upos_vocab, preprocessed_deprel_vocab, _, _, _, _ = \
                                                                    preprocess_data.load_preprocessed_data(args.load)

        form_vocab, upos_vocab, deprel_vocab, forms, upos, heads, deprels = preprocess_data.preprocess_data(
            file_path=args.input_file,
            form_vocab=preprocessed_form_vocab,
            upos_vocab=preprocessed_upos_vocab,
            deprel_vocab=preprocessed_deprel_vocab,
            update=args.update,
            max_len=args.max_len)

    if args.save:
        directory, filename = os.path.split(args.input_file)
        output_file = f"preprocessed_{filename}.pt"
        output_file = os.path.join(directory, output_file)

        preprocess_data.save_preprocessed_data(form_vocab, upos_vocab, deprel_vocab, 
                                               forms, upos, heads, deprels, output_file)

    if args.display:
        preprocess_data.display_preprocessed_data(form_vocab, upos_vocab, deprel_vocab, forms, upos, heads, deprels)


def handle_train(args):
    """Handles training mode"""

    if args.ltrain:
        form_vocab_train, upos_vocab_train, deprel_vocab_train, forms_train, upos_train, heads_train, deprels_train \
            = preprocess_data.load_preprocessed_data(args.ltrain)
    else:
        form_vocab_train, upos_vocab_train, deprel_vocab_train, forms_train, upos_train, heads_train, deprels_train \
            = preprocess_data.preprocess_data(file_path=args.ftrain)

    if args.ldev:
        form_vocab_dev, upos_vocab_dev, deprel_vocab_dev, forms_dev, upos_dev, heads_dev, deprels_dev \
            = preprocess_data.load_preprocessed_data(args.ldev)
    else:
        form_vocab_dev, upos_vocab_dev, deprel_vocab_dev, forms_dev, upos_dev, heads_dev, deprels_dev \
            = preprocess_data.preprocess_data(file_path=args.fdev, form_vocab=form_vocab_train, upos_vocab=upos_vocab_train, update=False)

    SemanticParser = biaffine_parser.biaffine_parser(V_w=len(form_vocab_train), V_t=len(upos_vocab_train),
                                                     V_l=len(deprel_vocab_train), d_w=args.d_w, d_t=args.d_t,
                                                     d_h=args.d_h, d_arc=args.d_arc, d_rel=args.d_rel,
                                                     rnn_model=args.rnn_type, rnn_layers=args.rnn_layers,
                                                     bidirectional=args.bidirectional, dropout_rate=args.dropout_rate)

    optimizer = torch.optim.Adam(SemanticParser.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()

    trainer = train_gbparser.Trainer(SemanticParser, optimizer, loss_function, args.batch_size)
    history = trainer.train(forms_train, upos_train, heads_train, deprels_train,
                            forms_dev, upos_dev, heads_dev, deprels_dev, args.n_epochs)

    if args.save and args.output is not None:
        tools.save_model(args.output, model=SemanticParser, optimizer=optimizer, criterion=loss_function,
                         trained_forms=form_vocab_train, trained_upos=upos_vocab_train, trained_deprels=deprel_vocab_train,
                         n_epochs=args.n_epochs, batch_size=args.batch_size, history=history)


def handle_predict(args):
    print(f"Loading model from {args.model}")
    model, optimizer, criterion, trained_forms, trained_upos, trained_deprels = \
        tools.load_model(trained_model_path=args.model, device=device)

    predict_gbparser.predict(model=model, input_file_path=args.input_file, output_file_path=args.output_file,
                             trained_forms=trained_forms, trained_upos=trained_upos, trained_deprels=trained_deprels,
                             display=args.display, device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Graph-based biaffine semantic parser of French")

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Select mode: preprocess | train | predict")
    parser.add_argument('--disable_cuda', action='store_true', help='disable CUDA')

    # --- Preprocessing Mode ---
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess data")
    preprocess_parser.add_argument("input_file", type=str, help="path to input file")
    preprocess_parser.add_argument("--load", '-l', type=str, help="path to preprocessed file")
    preprocess_parser.add_argument('--update', '-u', action='store_true',
                                   help='update vocabulary during preprocessing')
    preprocess_parser.add_argument('--max_len', '-m', type=int, default=50, help='maximum sequence length (default=50)')
    preprocess_parser.add_argument('--save', '-s', action='store_true', help='save preprocessed data')
    preprocess_parser.add_argument('--display', '-d', action='store_true', help='display preprocessed data')
    preprocess_parser.set_defaults(func=handle_preprocess)

    # --- Train Mode ---
    train_parser = subparsers.add_parser("train", help="Train model")

    train_parser.add_argument('--ftrain', type=str, default=cf.SEQUOIA_SIMPLE_TRAIN,
                              help="path to train corpus")
    train_parser.add_argument('--fdev', type=str, default=cf.SEQUOIA_SIMPLE_DEV, help="path to dev corpus")
    train_parser.add_argument('--ltrain', type=str, help="path to preprocessed train file")
    train_parser.add_argument('--ldev', type=str, help="path to preprocessed dev file")

    train_parser.add_argument('--d_w', type=int, default=100, help="dimension of form embeddings (default=100)")
    train_parser.add_argument('--d_t', type=int, default=100, help="dimension of upos embeddings (default=100)")
    train_parser.add_argument('--d_h', type=int, default=200, help="dimension of recurrent state (default=200)")
    train_parser.add_argument('--d_arc', type=int, default=400, help="dimension of head/dependent vector (default=400)")
    train_parser.add_argument('--d_rel', type=int, default=100, help="dimension of deprel vector (default=100)")
    train_parser.add_argument('--rnn_type', '-t', choices=['lstm', 'gru'], default='lstm',
                              help="type of rnn (default=lstm)")
    train_parser.add_argument('--rnn_layers', type=int, default=3, help="number of rnn's layer (default=3)")
    train_parser.add_argument('--bidirectional', action='store_true', help='enable bidirectional')
    train_parser.add_argument('--dropout_rate', '-r', type=float, default=0.33, help="dropout rate (default=0.33)")

    train_parser.add_argument('--n_epochs', type=int, default=10, help="number of training epochs")
    train_parser.add_argument('--batch_size', type=int, default=32, help="batch_size")

    train_parser.add_argument('--lr', type=float, default=0.002, help="learning rate")
    # TO DO : EARLY STOPPING TO BE IMPLEMENTED

    train_parser.add_argument('--save', '-s', action='store_true', help='save trained model')
    train_parser.add_argument('--output', '-o', type=str, default=None, help="path to save model")

    train_parser.set_defaults(func=handle_train)

    # --- Predict Mode ---
    predict_parser = subparsers.add_parser("predict", help="Predict semantic structures")

    predict_parser.add_argument('input_file', type=str, help="path to input file")
    predict_parser.add_argument('output_file', type=str, help="path to output file")
    predict_parser.add_argument('model', type=str, help="path to trained model")
    predict_parser.add_argument('--display', '-d', action='store_true', help='display the predictions')
    # predict_parser.add_argument('--trained_forms', type=str, default=None, help="path to trained form vocabulary")
    # predict_parser.add_argument('--trained_uposs', type=str, default=None, help="path to trained upos vocabulary")
    predict_parser.set_defaults(func=handle_predict)

    args = parser.parse_args()

    if args.disable_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f"Using {device}")

    args.func(args)
