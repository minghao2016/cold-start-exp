__author__ = 'shuochang'

import pandas as pd
import scipy as sp
import numpy as np


def df_to_matrix(df, col1, col2, value='value', col1_label_name=None, col2_label_name=None):
    if not (col1 in df.columns)&(col2 in df.columns):
        raise ValueError('Can not find the column in the data frame')
    if col1==col2:
        raise ValueError('Two dimensions can not be the same')
    unique_col1 = np.unique(df[col1])
    col1_id_map = pd.DataFrame({col1: unique_col1,
                                col1+'_idx': range(len(unique_col1))})
    unique_col2 = np.unique(df[col2])
    col2_id_map = pd.DataFrame({col2: unique_col2,
                                col2+'_idx': range(len(unique_col2))})
    df_idx = pd.merge(df, col1_id_map, left_on=col1, right_on=col1, how='inner')
    df_idx = pd.merge(df_idx, col2_id_map, left_on=col2, right_on=col2, how='inner')
    mat = sp.sparse.coo_matrix((df_idx[value],
                                (df_idx[col1+'_idx'], df_idx[col2+'_idx'])))
    if col1_label_name is None:
        col1_label_name = col1
    if col2_label_name is None:
        col2_label_name = col2
    row_label = df_idx[[col1+'_idx', col1_label_name]].drop_duplicates().set_index(col1+'_idx').sort()
    col_label = df_idx[[col2+'_idx', col2_label_name]].drop_duplicates().set_index(col2+'_idx').sort()
    return (df_idx, mat, row_label, col_label)


def mat_to_df(mat, label, movie):
    df = pd.DataFrame({'movie_x': mat.row,
                       'movie_y': mat.col,
                       'sim': mat.data})
    label_title = pd.merge(label.reset_index(), movie[['movieId', 'title']], on='movieId')
    df = pd.merge(df, label_title, left_on='movie_x', right_on='movieId_idx')
    df = pd.merge(df, label_title, left_on='movie_y', right_on='movieId_idx')
    return df

def movie_cluster(clust_label, movie_label, idx):
    return pd.merge(pd.DataFrame({'cluster':clust_label}).reset_index(),

                    movie_label.reset_index(), left_on='index', right_on=idx)


def path_sim(data_frames, half_path):
    trans_mat=[]
    length=len(half_path)
    for i in range(length-1):
        trans_names = set((half_path[i], half_path[i+1]))
        for df in data_frames:
            if trans_names.issubset(set(df.columns)):
                # transition matrix can be found in df
                _, mat, r_label, _ = df_to_matrix(df, half_path[i], half_path[i+1])
                trans_mat.append(mat.tocsr())
                if i==0:
                    row_label = r_label
    result_mat = trans_mat[0]
    for i in range(1, length-1):
        result_mat = result_mat * trans_mat[i]
    result_mat = result_mat * result_mat.T
    diag = result_mat.diagonal()
    row, col = result_mat.nonzero()
    data = result_mat.data
    new_data=[]
    for i in range(result_mat.nnz):
        new_data.append(2.0*data[i]/(diag[row[i]]+diag[col[i]]))
    final_mat = sp.sparse.coo_matrix((new_data, (row, col)))
    return (row_label, final_mat)


def get_output_name(fname, appendix):
    new_name = fname.split(".")
    new_name.insert(len(new_name)-1, appendix)
    return ".".join(new_name)


HASH = 100000

def fake_uid(clusters):
    return ((clusters+1)*HASH).astype(int)


def inverse_fake_uid(ids):
    return (ids/HASH-1).astype(int)
