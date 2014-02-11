__author__ = 'shuochang'

import argparse
import glob
from strategy import PopularityStrategy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_pattern', type=str, help='pattern of names for training files')
    parser.add_argument('test_pattern', type=str, help='pattern of names for testing files')
    args = parser.parse_args()

    train_files = glob.glob(args.train_pattern)
    test_files = glob.glob(args.test_pattern)
    if len(train_files) != len(test_files):
        raise ValueError('Number of training files not equal to number of testing files')

    popular = PopularityStrategy()
    for count, (train_n, test_n) in enumerate(zip(train_files, test_files)):
        print "Processing %d fold with cold start" % count
        popular.write_train_test(train_n, test_n, 10)

if __name__ == "__main__":
    main()