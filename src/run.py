import argparse

from prepare_data import *


def preprocess_data(args):
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
    preprocess_parser.set_defaults(func=preprocess_data)

    args = parser.parse_args()
    args.func(args)
