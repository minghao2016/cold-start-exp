__author__ = 'shuochang'

import argparse
import glob
import pandas as pd
import numpy as np
import cPickle as pickle
import sklearn as sk
from sklearn import metrics
from sklearn import cluster
from sklearn import decomposition
import utils
import warnings
import scipy as sp





class RatingCluster:

    def __init__(self):
        self.train_file = ''
        self.test_file = ''
        self.ratings_for_train = None
        self.ratings_from_test_user = None
        self.top_movies = None
        self.genome_dense = None
        self.rating_dense = None
        self.rating_dense_bias = None
        self.genome_dense = None

    def train(self, train_file, test_file, genome, movies):
        print "Initializing RatingCluster instance.."
        self.train_file = train_file
        self.test_file = test_file
        train_df = pd.read_csv(train_file,  header=None,
                               names=['userId', 'movieId', 'rating', 'timestamp'])
        train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], unit='s')
        test_df = pd.read_csv(test_file,  header=None,
                              names=['userId', 'movieId', 'rating', 'timestamp'])
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], unit='s')
        test_users = np.unique(test_df.userId)
        self.ratings_for_train = train_df[~train_df.userId.isin(test_users)]
        self.ratings_from_test_user = self.ratings_for_train[self.ratings_for_train.userId.isin(test_users)]
        grouped = self.ratings_for_train.groupby('movieId')
        movie_ratings = grouped.agg({'userId':'count',
                                     'rating':'mean'})
        last_time = np.max(self.ratings_for_train.timestamp)
        movie_ratings['span'] = grouped['timestamp'].agg(lambda x: (last_time - np.min(x)).days + 1).astype(int)
        movie_ratings.rename(columns={'userId':'num_ratings'}, inplace=True)
        top_movies = movie_ratings.sort('num_ratings', ascending=False)[0:200].reset_index()
        self.top_movies = pd.merge(top_movies, movies, on='movieId')

        top_ratings = self.ratings_for_train[self.ratings_for_train.movieId.isin(self.top_movies.movieId)]
        user_top_ratings = top_ratings.groupby('userId').agg({'movieId':'count',
                                                              'rating':'mean'}).reset_index() \
            .rename(columns={'movieId':'num_ratings'})
        active_user = user_top_ratings[user_top_ratings.num_ratings>150].userId
        self.rating_dense = self.ratings_for_train[(self.ratings_for_train.userId.isin(active_user))&
                                         (self.ratings_for_train.movieId.isin(self.top_movies.movieId))] \
            [['movieId', 'userId', 'rating']]
        self.rating_dense['value'] = self.rating_dense['rating']
        self.rating_dense_bias = self.rating_dense
        rating_dense_user_bias = self.rating_dense_bias.groupby('userId').agg({'value':'mean'}).reset_index()
        rating_dense_item_bias = self.rating_dense_bias.groupby('movieId').agg({'value':'mean'}).reset_index()
        self.rating_dense_bias = pd.merge(self.rating_dense_bias, rating_dense_user_bias,
                                          on='userId', suffixes=['', '_user'])
        self.rating_dense_bias = pd.merge(self.rating_dense_bias, rating_dense_item_bias,
                                          on='movieId', suffixes=['', '_movie'])
        self.rating_dense_bias['value_global'] = np.mean(self.rating_dense_bias.value)
        self.rating_dense_bias['value'] = self.rating_dense_bias['value'] - self.rating_dense_bias['value_user'] - \
                                     (self.rating_dense_bias['value_movie'] - self.rating_dense_bias['value_global'])
        self.genome_dense = genome[genome.movie_id.isin(self.top_movies.movieId)]
        self.genome_dense = self.genome_dense[self.genome_dense.relevance>3]
        self.genome_dense.rename(columns={'movie_id': 'movieId', 'relevance': 'value'}, inplace=1)

    def movie_cluster_spec(self, rating_mat, movie_idx, movies, K):
        rating_sim = 1-sk.metrics.pairwise_distances(rating_mat.toarray(), metric='cosine')
        # clamp the negative similarities
        rating_sim[rating_sim<0]=0
        spec_cluster = sk.cluster.spectral_clustering(rating_sim, n_clusters=K,
                                                      eigen_solver='arpack', assign_labels='kmeans')
        cluster_result = pd.merge(utils.movie_cluster(spec_cluster, movie_idx, 'movieId_idx'),
                                  movies[['movieId', 'title']], on='movieId')
        return cluster_result

    def movie_cluster_svd_spec(self, rating_mat, movie_idx, movies, K):
        svd = sk.decomposition.PCA(n_components=10, whiten=True)
        movie_feat = svd.fit_transform(rating_mat.toarray())
        rating_sim = 1-sk.metrics.pairwise_distances(movie_feat, metric='cosine')
        # clamp the negative similarities
        rating_sim[rating_sim<0]=0
        spec_cluster = sk.cluster.spectral_clustering(rating_sim, n_clusters=K,
                                                      eigen_solver='arpack', assign_labels='kmeans')
        cluster_result = pd.merge(utils.movie_cluster(spec_cluster, movie_idx, 'movieId_idx'),
                                  movies[['movieId', 'title']], on='movieId')
        return cluster_result

    def clustering(self, algorithms, Ks):
        _, rating_dense_mat, mat_movie_idx, _ = \
            utils.df_to_matrix(self.rating_dense, col1='movieId', col2='userId')
        _, rating_dense_bias_mat, bias_mat_movie_idx, _ = \
            utils.df_to_matrix(self.rating_dense_bias, col1='movieId', col2='userId')
        _, tag_dense_mat, mat_tag_idx, _ = \
            utils.df_to_matrix(self.genome_dense, col1='movieId', col2='tag')
        unique_algos = np.unique(algorithms)
        for k in Ks:
            for algo in unique_algos:
                print "Clustering using %s into %s clusters" % (algo, str(k))
                if algo=='spectral':
                    cluster_res = self.movie_cluster_spec(rating_dense_mat, mat_movie_idx,
                                                          self.top_movies, k)
                    cluster_res_bias = self.movie_cluster_spec(rating_dense_bias_mat, mat_movie_idx,
                                                            self.top_movies, k)
                elif algo=='spectral_svd':
                    cluster_res = self.movie_cluster_svd_spec(rating_dense_mat, mat_movie_idx,
                                                          self.top_movies, k)
                    cluster_res_bias = self.movie_cluster_svd_spec(rating_dense_bias_mat, mat_movie_idx,
                                                               self.top_movies, k)
                else:
                    warnings.warn('Unrecogized algorithm: %s' % algo)
                    continue
                name = algo+'_'+str(k)
                self.pseudo_cluster_tt(cluster_res, 'original', name)
                self.pseudo_cluster_tt(cluster_res_bias, 'bias', name)

    def pseudo_cluster_tt(self, cluster, type, name):
        if type == 'bias':
            rating_cluster = pd.merge(self.rating_dense_bias, cluster, on = 'movieId')
        elif type == 'original':
            rating_cluster = pd.merge(self.rating_dense, cluster, on = 'movieId')
        else:
            raise ValueError('Unrecoginzed type')
        rating_cluster = rating_cluster.groupby(['userId', 'cluster'])['value'].agg(np.mean).reset_index()
        user_fav_cluster = rating_cluster.groupby('userId').apply(lambda x: x.loc[x['value']
                                                                  .argmax()]).reset_index(drop=1)
        tmp = pd.merge(self.rating_dense, user_fav_cluster, on='userId')
        pseudo_rating = tmp.groupby(['cluster', 'movieId'])['rating'].agg(np.mean).reset_index()
        pseudo_rating['userId'] = utils.fake_uid(pseudo_rating.cluster)

        train_ratings = pd.concat([self.ratings_for_train[['userId', 'movieId', 'rating']],
                                   pseudo_rating[['userId', 'movieId', 'rating']]],
                                  ignore_index =True)
        train_ratings.to_csv(utils.get_output_name(self.train_file, type+'_'+name),
                             header=False, index=False)
        test_ratings = pd.DataFrame({'userId': np.unique(pseudo_rating.userId), 'movieId': 1, 'rating': 2})
        test_ratings[['userId', 'movieId', 'rating']]\
            .to_csv(utils.get_output_name(self.test_file, type+'_'+name), header=False, index=False)
        labels = self.label_clusters(cluster)
        labels.to_csv(utils.get_output_name(self.train_file, type+'_'+name+'_label'),
                      index=False)
        cluster.to_csv(utils.get_output_name(self.train_file, type+'_'+name+'_cluster'),
                       index=False)
        pseudo_rating[['userId', 'movieId', 'rating']].to_csv(
            utils.get_output_name(self.train_file, type+'_'+name+'_user'), index=False)

    def label_clusters(self, cluster):
        tag_dense = self.genome_dense
        tag_cluster = pd.merge(cluster, tag_dense, on='movieId')
        tag_cluster_agg = tag_cluster.groupby(['cluster', 'tag']).agg({'value': 'sum'}) \
            .reset_index()
        tmp_cluster_sum = tag_cluster_agg.groupby('cluster').agg(np.sum).reset_index()
        tmp_tag_sum = tag_cluster_agg.groupby('tag')[['value']].agg(np.sum).reset_index()
        tag_cluster_agg = pd.merge(tag_cluster_agg, tmp_cluster_sum, on='cluster',
                                   suffixes=['', '_cluster_sum'])
        tag_cluster_agg = pd.merge(tag_cluster_agg, tmp_tag_sum, on='tag', suffixes=['', '_tag_sum'])
        tag_cluster_agg['relevance'] = tag_cluster_agg.value*1.0/tag_cluster_agg.value_cluster_sum
        tag_cluster_agg['distinct'] = tag_cluster_agg.value*1.0/tag_cluster_agg.value_tag_sum
        tag_cluster_agg['utility'] = np.power(tag_cluster_agg['distinct'] \
                                              *tag_cluster_agg['relevance'],1.0/2)
        tag_cluster_agg.sort(['cluster','utility'], ascending=[1, 0], inplace=1)
        tag_labels = tag_cluster_agg.groupby('cluster').apply(lambda x:x[:3]).reset_index(drop=1)
        movie_tag_cluster = pd.merge(tag_labels[['cluster', 'tag']], tag_cluster, on='tag')
        movie_tag_cluster= movie_tag_cluster[movie_tag_cluster.cluster_x==movie_tag_cluster.cluster_y]
        tmp_cluster_movie = movie_tag_cluster.groupby(['title','cluster_x'])['value'].agg(np.sum).reset_index()
        tmp_cluster_movie.sort(['cluster_x', 'value'], ascending=[1,0], inplace=1)
        movie_labels = tmp_cluster_movie.groupby('cluster_x').apply(lambda x:x[:3]) \
            .reset_index(drop=1).rename(columns={'cluster_x':'cluster'})
        labels = pd.merge(movie_labels, tag_labels, on='cluster')
        return labels



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_pattern', type=str, help='pattern of names for training files')
    parser.add_argument('test_pattern', type=str, help='pattern of names for testing files')
    parser.add_argument('genome_file', type=str, help='location of genome file')
    parser.add_argument('movie_file', type=str, help='location of movie file')
    parser.add_argument('--algorithms', type=str, nargs='+')
    parser.add_argument('--k', type=int, nargs='+')
    args = parser.parse_args()

    train_files = glob.glob(args.train_pattern)
    test_files = glob.glob(args.test_pattern)
    with open(args.genome_file) as f:
        genome = pickle.load(f)
    with open(args.movie_file) as f:
        movies = pickle.load(f)

    if len(train_files) != len(test_files):
        raise ValueError('Number of training files not equal to number of testing files')

    for count, (train_n, test_n) in enumerate(zip(train_files, test_files)):
        print "Processing %d fold with cluster rating" % count
        rating_cluster = RatingCluster()
        rating_cluster.train(train_n, test_n, genome, movies)
        rating_cluster.clustering(args.algorithms, args.k)


if __name__ == "__main__":
    main()