import argparse

from prepare_data import *
from biaffine_parser import *
from src.train_gbparser import Trainer


def handle_preprocess(args):
    """Handles data preprocessing mode"""
    print(f"Preprocessing file; {args.input_file}")

    if args.load is None:
        word_vocab, tag_vocab, words, tags, governors = prepare_data(file_path=args.input_file,
                                                                     update=args.update,
                                                                     max_len=args.max_len)
    else:
        preprocessed_word_vocab, preprocessed_tag_vocab, _, _, _ = load_preprocessed_data(args.load)
        word_vocab, tag_vocab, words, tags, governors = prepare_data(file_path=args.input_file,
                                                                     word_vocab=preprocessed_word_vocab,
                                                                     tag_vocab=preprocessed_tag_vocab,
                                                                     update=args.update,
                                                                     max_len=args.max_len)

    if args.save:
        directory, filename = os.path.split(args.input_file)
        output_file = f"preprocessed_{filename}.pt"
        output_file = os.path.join(directory, output_file)

        save_preprocessed_data(word_vocab, tag_vocab, words, tags, governors, output_file)

    if args.display:
        display_preprocessed_data(word_vocab, tag_vocab, words, tags, governors)


def handle_train(args):
    """Handles training mode"""
    word_vocab_train, tag_vocab_train, words_train, tags_train, governors_train = prepare_data(file_path=args.ftrain)

    word_vocab_dev, tag_vocab_dev, words_dev, tags_dev, governors_dev = prepare_data(file_path=args.fdev,
                                                                                     word_vocab=word_vocab_train,
                                                                                     tag_vocab=tag_vocab_train,
                                                                                     update=False)

    BiAffineParser = biaffine_parser(len(word_vocab_train), len(tag_vocab_train), args.d_w, args.d_t, args.d_h, args.d)

    optimizer = torch.optim.Adam(BiAffineParser.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()

    trainer = Trainer(BiAffineParser, optimizer, loss_function, 32)
    history = trainer.train(words_train, tags_train, governors_train, words_dev, tags_dev, governors_dev, args.n_epochs)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Graph-based biaffine semantic parser of French")

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Select mode: preprocess | train | predict")

    # --- Preprocessing Mode ---
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess data")
    preprocess_parser.add_argument("input_file", type=str, help="path to input file")
    preprocess_parser.add_argument("--load", '-l', type=str, help="path to preprocessed file")
    preprocess_parser.add_argument('--update', '-u', action='store_true',
                                   help='update vocabulary during preprocessing')
    preprocess_parser.add_argument('--max_len', '-m', type=int, default=30, help='maximum sequence length (default=30)')
    preprocess_parser.add_argument('--save', '-s', action='store_true', help='save preprocessed data')
    preprocess_parser.add_argument('--display', '-d', action='store_true', help='display preprocessed data')
    preprocess_parser.set_defaults(func=handle_preprocess)

    # --- Train Mode ---
    train_parser = subparsers.add_parser("train", help="Train model")

    train_parser.add_argument('--ftrain', type=str, default=config.SEQUOIA_SIMPLE_TRAIN, help="path to train corpus")
    train_parser.add_argument('--fdev', type=str, default=config.SEQUOIA_SIMPLE_DEV, help="path to dev corpus")
    train_parser.add_argument('--ltrain', type=str, help="path to preprocessed train file")
    train_parser.add_argument('--ldev', type=str, help="path to preprocessed dev file")

    train_parser.add_argument('--d_w', type=int, default=100, help="dimension of word embeddings")
    train_parser.add_argument('--d_t', type=int, default=100, help="dimension of tag embeddings")
    train_parser.add_argument('--d_h', type=int, default=200, help="dimension of recurrent state")
    train_parser.add_argument('--d', type=int, default=400, help="dimension of head/dependent vector")

    train_parser.add_argument('--n_epochs', type=int, default=30, help="number of training epochs")

    train_parser.add_argument('--lr', type=float, default=0.001, help="learning rate")

    train_parser.add_argument('--save', '-s', action='store_true', help='save trained model')

    train_parser.set_defaults(func=handle_train)

    args = parser.parse_args()
    args.func(args)
