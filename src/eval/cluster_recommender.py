__author__ = 'shuochang'


import argparse
import pandas as pd
import numpy as np
import utils
import sys
import warnings


class ClusterRecommender():
    def __init__(self):
        self.prediction_name = ''
        self.cluster_name = ''
        self.recommend_df = None
        self.model = None
        self.item_set = {}
        self.user_cluster_map = {}
        self.cluster_df = None
        self.partition = 0

    def train(self, cluster_name, prediction_name, recommend_file, cluster_file, partition):
        self.partition = partition
        self.cluster_name = cluster_name
        self.prediction_name = prediction_name
        self.recommend_df = pd.read_csv(recommend_file)
        self.model = self.recommend_df[(self.recommend_df.Algorithm == self.prediction_name) &
                                  (self.recommend_df.DataSet == self.cluster_name) &
                                    (self.recommend_df.Partition == self.partition)]
        self.model['cluster'] = utils.inverse_fake_uid(self.model.User)
        if self.model.shape[0] == 0:
            raise ValueError('Can not find predictions from recommend output file')
        self.item_set = pd.unique(self.model.Item)
        self.cluster_df = pd.read_csv(cluster_file)

    def score_item(self, train_file, test_file, score_type):
        train_df = pd.read_csv(train_file, header=False,
                               names=['userId', 'movieId', 'rating', 'timestamp'])
        test_df = pd.read_csv(test_file, header=False,
                              names=['userId', 'movieId', 'rating', 'timestamp'])
        test_users = pd.unique(test_df.userId)
        rating_from_test_users = train_df[train_df.userId.isin(test_users)]
        rating_from_test_users_cluster = pd.merge(self.cluster_df, rating_from_test_users,
                                                  on='movieId')
        rating_from_test_users_cluster = rating_from_test_users_cluster.groupby(['userId', 'cluster'])['rating'] \
                                    .agg(np.mean).reset_index()
        if score_type == 'optimal':
            rating_join = pd.merge(test_df, self.model,
                                   left_on='movieId', right_on='Item', how='left') \
                .drop(['timestamp', 'Partition', 'Item', 'Rank'], axis=1)
            rating_join['error'] = rating_join['rating'] - rating_join['Score']
            rmse_summary = rating_join.groupby(['userId', 'cluster'])['error'] \
                    .agg(lambda x: np.linalg.norm(x)/np.sqrt(len(x))).reset_index()
            test_user_cluster_map = rmse_summary.groupby('userId') \
                .apply(lambda x: x.loc[x['error'].argmin()]).reset_index(drop=1)
        else:
            test_user_cluster_map = rating_from_test_users_cluster.groupby('userId') \
                .apply(lambda x: x.loc[x['rating'].argmax()]).reset_index(drop=1)
        output = pd.merge(test_user_cluster_map, self.model, on='cluster', how='left')
        output.userId = output.userId.astype(np.int64)
        output.Item = output.Item.astype(np.int64)
        output[['userId', 'Item', 'Score']].to_csv(sys.stdout, header=False, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', type=str, help='Location of training files')
    parser.add_argument('test_file', type=str, help='Location of testing files')
    parser.add_argument('model_file', type=str, help='Location of the model recommendation file')
    parser.add_argument('cluster_name', type=str,
                        help='Name of the clustering algorithm backing the recommendation')
    parser.add_argument('prediction', type=str,
                        help='The list of prediction algorithm names')
    parser.add_argument('partition', type=int,
                        help='The number of the file partition')
    parser.add_argument('score_type', type=str, help='Type of scoring process')
    args = parser.parse_args()

    recommender = ClusterRecommender()
    cluster_file = utils.get_output_name(args.train_file, args.cluster_name+'_cluster')
    recommender.train(args.cluster_name, args.prediction, args.model_file, cluster_file, args.partition)
    if args.score_type not in ['optimal', 'simulation']:
        raise ValueError('Unrecognized input for score_type')
    recommender.score_item(args.train_file, args.test_file, args.score_type)




if __name__=='__main__':
    main()