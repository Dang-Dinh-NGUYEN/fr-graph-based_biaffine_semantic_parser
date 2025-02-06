import argparse

import config
from prepare_data import prepare_data, load_preprocessed_data, save_preprocessed_data, display_preprocessed_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Graph-based biaffine semantic parser of French")

    parser.add_argument('mode', type=str, choices=['preprocess', 'train', 'predict'])

    # Data arguments
    data = parser.add_argument_group('Data Options')
    data.add_argument('--ftrain', type=str, default=config.SEQUOIA_SIMPLE_TRAIN, help='path to train corpus')
    data.add_argument('--fdev', type=str, default=config.SEQUOIA_SIMPLE_DEV, help='path to dev corpus')
    data.add_argument('--ftest', type=str, default=config.SEQUOIA_SIMPLE_TEST, help='path to test corpus')

    data.add_argument('--save', '-s', action='store_true', help='save preprocessed data')

    data.add_argument('--load', '-l', action='store_true', help='load preprocessed data')
    data.add_argument('--preprocessed', type=str, default=None, help='path to preprocessed data')

    data.add_argument('--display', '-d', action='store_true', help='display preprocessed data')

    args = parser.parse_args()

    if args.mode == 'preprocess':
        if not args.load:
            word_vocab, tag_vocab, words, tags, governors = prepare_data(args.ftrain)

        else:
            if args.preprocessed is None:
                parser.error("--load (-l) requires --preprocessed to specify the path of preprocessed data.")
            word_vocab, tag_vocab, words, tags, governors = load_preprocessed_data(args.preprocessed)

        if args.save:
            if args.preprocessed is None:
                parser.error("--save (-s) requires --preprocessed to specify the output path.")

            save_preprocessed_data(word_vocab, tag_vocab, words, tags, governors, args.preprocessed)

        if args.display:
            display_preprocessed_data(word_vocab, tag_vocab, words, tags, governors)


