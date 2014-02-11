__author__ = 'shuochang'

import abc
import pandas as pd


class BaseColdStartStrategy(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def gen_movie_list(self, train_df, n):
        """generate a sequence of movies to rate"""

    @abc.abstractmethod
    def write_train_test(self, train_name, test_name, n):
        """prepare the train and test files after applying the cold start strategy"""

    def gen_name(self, name):
        new_name = name.split(".")
        new_name.insert(len(new_name)-1, self.name)
        return ".".join(new_name)


class PopularityStrategy(BaseColdStartStrategy):
    def __init__(self, name='popular'):
        self.name = name

    def gen_movie_list(self, train_df, n):
        num_rating = train_df.groupby('item')['rating'].count()
        num_rating.sort(ascending=False)
        return num_rating.index[:n]

    def write_train_test(self, train_name, test_name, n):
        train = pd.read_csv(train_name, header=None, names=['user', 'item', 'rating', 'time'])
        test = pd.read_csv(test_name, header=None, names=['user', 'item', 'rating', 'time'])
        test_ids = test.user.unique()
        user_fit = train.user.isin(test_ids)
        train_other_folds = train[~user_fit]
        train_this_fold = train[user_fit]
        # generate movie list and select only these to include in the training part
        # for this fold and combine with the other folds to make final training file
        movie_list = self.gen_movie_list(train_other_folds, n)
        train_selected = train_this_fold[train_this_fold.item.isin(movie_list)]
        train_final = pd.concat([train_other_folds, train_selected])
        # output to files
        train_final.to_csv(self.gen_name(train_name), header=False)
        test.to_csv(self.gen_name(test_name), header=False)
