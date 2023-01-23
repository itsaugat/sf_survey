import pandas as pd
import numpy as np
from sklearn.compose import make_column_selector as selector


'''
get training data stats
'''
def get_numerical_std(dataset, target_col):
    
    # ignore target column
    dataset_sub = dataset.loc[:, dataset.columns != target_col]
    # select numerical and categorical data 
    numerical_columns_selector = selector(dtype_exclude=object)
    numerical_columns = numerical_columns_selector(dataset_sub)
    
    std_dict = {}
    
    for col in numerical_columns:
        std_dict[dataset.columns.get_loc(col)] = dataset[col].std()
    
    return std_dict


'''
compute tolerance level to check equality of numerical feature values
20% of the std 
'''
def num_tolerance(std):
    level = 0.2
    tol = level * std
    return tol



'''
compute number of common numerical features
'''
def count_num_com(num_idx, mdn, query, key_feat_idx, std_dict):
    count = 0
    # ignore the key feature in consideration
    num_idx_diff = set(num_idx) - set([key_feat_idx])
    for idx in list(num_idx_diff):
        # get std of the feature
        std = std_dict[idx]
        # get tolerance level for the feature
        tol = num_tolerance(std)
        # check if the query value lies within the range of tolerance
        if mdn[idx]+tol > mdn[idx]:
            if float(mdn[idx]-tol) <= float(query[idx]) <= float(mdn[idx]+tol):
                count += 1
        elif mdn[idx]+tol < mdn[idx]:
            if float(mdn[idx]+tol) <= float(query[idx]) <= float(mdn[idx]-tol):
                count += 1
    return count



'''
compute number of common categorical features
'''
def count_cat_com(cat_idx, mdn, query, key_feat_idx):
    count = 0
    # ignore the key feature in consideration
    cat_idx_diff = set(cat_idx) - set([key_feat_idx])
    for idx in list(cat_idx_diff):
        if mdn[idx] == query[idx]:
            count += 1

    return count



'''
compute semi-factual value if the key feature is continuous
'''
def compute_sf_value_continuous(tot_com_count, feat_diff, max_feat_diff, total_feats):
    # compute the overall value
    #val = (tot_com_count / total_feats) + (feat_diff / max_feat_diff)
    #return val

    # compute the overall value
    if(max_feat_diff == 0):
        val = (tot_com_count / total_feats)
    else:
        val = (tot_com_count / total_feats) + (feat_diff / max_feat_diff)

    return val



'''
compute semi-factual value if the key feature is categorical
'''
def compute_sf_value_categorical(tot_com_count, feat_diff, max_feat_diff, total_feats):
    # compute the overall value
    m1 = (tot_com_count / total_feats) 
    m2 = (feat_diff / max_feat_diff)
    total = m1 + m2

    return total


'''
Nearest Hit -> Instance that is most similar to query (Q) and has the same class as Q
'''
def get_nearest_hit_index(pred, indices, y_train):
    for index in indices[0]:
        # if the actual label of the query is same as the prediction then return it as the nearest hit
        if(y_train[index] == pred):
            return index
        


        
'''
NMOTB -> Nearest miss that lies over the decision boundary.
1. has different class from Q
2. most similar to Q
3. is not located, according to similarity, between Q and NH [Sim(Q,NH) > Sim(NMOTB,NH)]
'''
def get_nmotb_index(pred, indices, distances, nearest_hit_idx, y_train):
    # get distance between Q and NH
    dist_q_nh = distances[0][nearest_hit_idx]
    # loop through indices
    for index in indices[0]:
        if(y_train[index] != pred):
            # get distance between neighbor and NH
            dist_n_nh = distances[0][index]
            if dist_q_nh < dist_n_nh:
                return index

            
            
'''
Calculate k-ratio ->  (num of k from sf to nh / num of k from sf to nmotb)
Higher the value -> better the sf computed
Parameters : 
nh_idx -> index of nearest hit instance
nmotb_idx -> index of nmotb instance
nbrs -> nearest neighbor model fitted on training data
'''
def calculate_k(nearest_hit_idx, nmotb_idx, nbrs, sf):
    # get distances and indices for neighbors of the computed SF
    distances, indices = nbrs.kneighbors(np.reshape(sf, (1, -1)))
    # get the index of query_idx in the indices list and add 1 to get the value of k
    num_k_sf_nh = indices[0].tolist().index(nearest_hit_idx) + 1
    # get the index of nmotb_idx in the indices list and add 1 to get the value of k
    num_k_sf_nmotb = indices[0].tolist().index(nmotb_idx) + 1
    # calculate the ratio
    #k_ratio = float(num_k_sf_nh / num_k_sf_nmotb)
    return num_k_sf_nh, num_k_sf_nmotb


