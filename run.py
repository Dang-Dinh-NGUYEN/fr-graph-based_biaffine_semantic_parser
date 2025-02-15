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
        word_vocab, tag_vocab, label_vocab, words, tags, governors, deprels = \
                    preprocess_data.preprocess_data(file_path=args.input_file, update=args.update, max_len=args.max_len)
    else:
        preprocessed_word_vocab, preprocessed_tag_vocab, preprocessed_label, _, _, _, _ = \
                                                                    preprocess_data.load_preprocessed_data(args.load)

        word_vocab, tag_vocab, label_vocab, words, tags, governors, deprels = preprocess_data.preprocess_data(
            file_path=args.input_file,
            word_vocab=preprocessed_word_vocab,
            tag_vocab=preprocessed_tag_vocab,
            label_vocab=preprocessed_label,
            update=args.update,
            max_len=args.max_len)

    if args.save:
        directory, filename = os.path.split(args.input_file)
        output_file = f"preprocessed_{filename}.pt"
        output_file = os.path.join(directory, output_file)

        preprocess_data.save_preprocessed_data(word_vocab, tag_vocab, label_vocab, words, tags, governors, deprels, output_file)

    if args.display:
        preprocess_data.display_preprocessed_data(word_vocab, tag_vocab, label_vocab, words, tags, governors, deprels)


def handle_train(args):
    """Handles training mode"""

    if args.ltrain:
        word_vocab_train, tag_vocab_train, label_vocab_train, words_train, tags_train, governors_train, deprels_train = preprocess_data.load_preprocessed_data(
            args.ltrain)
    else:
        word_vocab_train, tag_vocab_train, label_vocab_train, words_train, tags_train, governors_train, deprels_train = preprocess_data.preprocess_data(
            file_path=args.ftrain, update=True, max_len=50)

    if args.ldev:
        word_vocab_dev, tag_vocab_dev, label_vocab_dev, words_dev, tags_dev, governors_dev, deprels_dev = preprocess_data.load_preprocessed_data(
            args.ldev)
    else:
        word_vocab_dev, tag_vocab_dev, label_vocab_dev, words_dev, tags_dev, governors_dev, deprels_dev = preprocess_data.preprocess_data(
            file_path=args.fdev, word_vocab=word_vocab_train, tag_vocab=tag_vocab_train, update=False, max_len=50)

    SemanticParser = biaffine_parser.biaffine_parser(V_w=len(word_vocab_train), V_t=len(tag_vocab_train),
                                                     V_l=len(label_vocab_train), d_w=args.d_w, d_t=args.d_t,
                                                     d_h=args.d_h, d_arc=args.d_arc, d_rel=args.d_rel,
                                                     dropout_rate=args.dropout_rate)

    optimizer = torch.optim.Adam(SemanticParser.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()

    trainer = train_gbparser.Trainer(SemanticParser, optimizer, loss_function, args.batch_size)
    history = trainer.train(words_train, tags_train, governors_train, deprels_train, words_dev, tags_dev, governors_dev, deprels_dev, args.n_epochs)

    if args.save and args.output is not None:
        tools.save_model(args.output, model=SemanticParser, optimizer=optimizer, criterion=loss_function,
                         trained_words=word_vocab_train, trained_tags=tag_vocab_train, trained_labels=label_vocab_train, n_epochs=args.n_epochs,
                         batch_size=args.batch_size, history=history)


def handle_predict(args):
    print(f"Loading model from {args.model}")
    model, optimizer, criterion, trained_words, trained_tags, trained_labels = \
        tools.load_model(trained_model_path=args.model, device=device)

    predict_gbparser.predict(model=model, input_file_path=args.input_file, output_file_path=args.output_file,
                             trained_words=trained_words, trained_tags=trained_tags, trained_label=trained_labels, device=device)


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
    preprocess_parser.add_argument('--max_len', '-m', type=int, default=50, help='maximum sequence length (default=30)')
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

    train_parser.add_argument('--d_w', type=int, default=100, help="dimension of word embeddings")
    train_parser.add_argument('--d_t', type=int, default=100, help="dimension of tag embeddings")
    train_parser.add_argument('--d_h', type=int, default=200, help="dimension of recurrent state")
    train_parser.add_argument('--d_arc', type=int, default=400, help="dimension of head/dependent vector")
    train_parser.add_argument('--d_rel', type=int, default=100, help="dimension of label vector")
    train_parser.add_argument('--dropout_rate', '-r', type=float, default=0.33, help="dropout rate")

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
    # predict_parser.add_argument('--trained_words', type=str, default=None, help="path to trained word vocabulary")
    # predict_parser.add_argument('--trained_tags', type=str, default=None, help="path to trained tag vocabulary")
    predict_parser.set_defaults(func=handle_predict)

    args = parser.parse_args()

    if args.disable_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f"Using {device}")

    args.func(args)
