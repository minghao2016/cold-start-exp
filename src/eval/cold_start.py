__author__ = 'shuochang'

import argparse
import glob
from strategy import PopularityStrategy, EntropyStrategy, EntropyZeroStrategy
import pandas as pd


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
    entropy = EntropyStrategy()
    entropy_zero = EntropyZeroStrategy()
    nums = [5, 10, 15]
    for count, (train_n, test_n) in enumerate(zip(train_files, test_files)):
        print "Processing %d fold with cold start" % count
        train = pd.read_csv(train_n, header=None, names=['user', 'item', 'rating', 'time'])
        test = pd.read_csv(test_n, header=None, names=['user', 'item', 'rating', 'time'])
        test_ids = test.user.unique()
        user_fit = train.user.isin(test_ids)
        train_other_folds = train[~user_fit]
        train_this_fold = train[user_fit]
        for n in nums:
            select_fn = 'rated'
            list_fn = 'list'
            popular.write_train_test(train_n, test_n, select_fn, list_fn,
                                     train_this_fold, train_other_folds, test, n)
            entropy.write_train_test(train_n, test_n, select_fn, list_fn,
                                     train_this_fold, train_other_folds, test, n)
            entropy_zero.write_train_test(train_n, test_n, select_fn, list_fn,
                                          train_this_fold, train_other_folds, test, n)

if __name__ == "__main__":
    main()