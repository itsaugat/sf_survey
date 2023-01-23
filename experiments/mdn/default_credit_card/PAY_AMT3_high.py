import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import NearestNeighbors
from cat_to_num import map_cat_to_num, cat_num_embed
from most_distant_neighbor import get_most_distant_neighbor
from helpers import get_numerical_std, num_tolerance, count_num_com, count_cat_com, compute_sf_value_continuous, compute_sf_value_categorical, get_nearest_hit_index, get_nmotb_index, calculate_k
from metrics import calculate_l1_distance, calculate_l2_distance, calculate_mahalanobis_dist
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm
import pickle
#import joblib
#from joblib import Parallel, delayed
#import multiprocess as mp
import warnings
warnings.simplefilter('ignore', UserWarning)




def sf_loop(x, y, train_index, test_index, dataset_embed, cat_embed, feat_order, cat_cols, cat_idx,
            num_cols, num_idx, num_classes, target, col_types, distance_metric,
            key_feature, key_feature_type, mdn_type, k_neighbors, std_dict, transformer, key_feat_idx):

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # y_train.reset_index(inplace=True, drop=True), y_test.reset_index(inplace=True, drop=True)

    #     start = time.time()
    #     # create categorical embedded dataframe for x_train (to get MDN from the x_train)
    #     train = np.column_stack([x_train, y_train])
    #     print(time.time() - start)
    #     # convert to df
    #     start = time.time()
    #     train_df = pd.DataFrame(train, columns = feat_order).astype(dtype = col_types)
    #     train_df_embed = map_cat_to_num(train_df, target, distance_metric)
    #     print(time.time() - start)

    ############################## initial steps for sf evaluation #####################
    # normalize categorical feature values
    # start = time.time()
    for x in x_train:
        for idx in cat_idx:
            x[idx] = float(cat_embed[idx][x[idx]])

    # convert numpy array to df for normalizing numerical features
    df = pd.DataFrame(x_train, columns=feat_order[:-1])
    # normalize numerical feature values
    df[num_cols] = transformer.fit_transform(df[num_cols])
    # convert df to numpy array
    x_train = df.to_numpy()
    # print(time.time() - start)

    # start = time.time()
    # get all the neigbors
    num_neighbors = len(train_index)
    # fit nearest neighbors on training data
    nbrs = NearestNeighbors(n_neighbors=num_neighbors).fit(x_train)
    # print(time.time() - start)

    # fit trust score on training data
    # ts.fit(x_train, y_train, classes=num_classes)

    ################################### compute native sf ######################

    query = x_test[0]

    # convert into dictionary
    row = {}
    for a, b in zip(feat_order[:-1], query):
        row[a] = b

    # add target column to query at the end
    row[target] = y_test[0]

    # make a copy of the query to use it later
    row_cp = row.copy()
    # get the list of mdn's
    # start = time.time()
    mdn_list = get_most_distant_neighbor(row_cp,
                                         target,
                                         dataset_embed,  # finds mdn in the whole dataset (also contains query)
                                         cat_cols,
                                         cat_idx,
                                         num_cols,
                                         num_idx,
                                         cat_embed,
                                         distance_metric,
                                         key_feature,
                                         mdn_type)

    # print(time.time() - start)
    row.popitem()  # remove the target value from the row assuming it is at the end
    row = list(row.values())
    # print(row)

    # compute the native_sf for the row
    sf_list = []
    count_list = []

    # get total features
    total_feats = len(num_idx) + len(cat_idx)
    # start = time.time()
    for mdn in mdn_list:
        mdn = mdn[0]
        cat_count = count_cat_com(cat_idx, mdn, row, key_feat_idx)
        num_count = count_num_com(num_idx, mdn, row, key_feat_idx, std_dict)
        tot_com_count = num_count + cat_count
        toal_feat_diff = total_feats - tot_com_count
        # com_ft_ratio = tot_com_count / (len(num_idx) + len(cat_idx))

        if key_feature_type == 'continuous':
            max_feat_diff = abs(mdn_list[0][0][key_feat_idx] - row[key_feat_idx])
            if (max_feat_diff == 0):
                max_feat_diff = row[key_feat_idx]
            feat_diff = abs(mdn[key_feat_idx] - row[key_feat_idx])

            sf_val = compute_sf_value_continuous(tot_com_count, feat_diff, max_feat_diff, total_feats)

        elif key_feature_type == 'categorical':
            # encode max mdn
            max_mdn_encode = mdn_list[0][0].copy()
            for idx in cat_idx:
                max_mdn_encode[idx] = float(cat_embed[idx][max_mdn_encode[idx]])

            # encode categorical features of mdn
            mdn_encode = mdn.copy()
            for idx in cat_idx:
                mdn_encode[idx] = float(cat_embed[idx][mdn_encode[idx]])

            # encode query
            query_encode = query.copy()
            for idx in cat_idx:
                query_encode[idx] = float(cat_embed[idx][query_encode[idx]])

            max_feat_diff = abs(max_mdn_encode[key_feat_idx] - query_encode[key_feat_idx])

            if (max_feat_diff == 0):
                max_feat_diff = query_encode[key_feat_idx]
            feat_diff = abs(mdn_encode[key_feat_idx] - query_encode[key_feat_idx])

            sf_val = compute_sf_value_categorical(tot_com_count, feat_diff, max_feat_diff, total_feats)

        sf_list.append(sf_val)
        count_list.append(toal_feat_diff)

    # print(time.time() - start)

    # get the native_sf based on the value computed
    sf_val = max(sf_list)
    # get the index of native_sf
    sf_idx = sf_list.index(sf_val)
    # get the native_sf
    sf = mdn_list[sf_idx]
    # print(sf)

    # get the number of dissimilar features in the native_sf
    feat_list = count_list[sf_idx]

    ########################## evaluation metrics #################################

    # normalize sf
    sf = np.array(sf[0])
    # normalize numerical features of sf
    sf[num_idx] = transformer.transform(np.reshape(sf[num_idx], (1, -1)))
    # normalize categorical variables of sf
    for idx in cat_idx:
        sf[idx] = float(cat_embed[idx][sf[idx]])
    # convert sf to float array
    sf = sf.astype(float)
    # print(sf)

    # reshape query instance
    # query_sc = np.array(query)
    # normalize numerical variables of query instance
    query[num_idx] = transformer.transform(np.reshape(query[num_idx], (1, -1)))
    # normalize categorical variables of query instance
    for idx in cat_idx:
        query[idx] = float(cat_embed[idx][query[idx]])
    # convert query to float array
    query_sc = query.astype(float)
    query_sc = np.reshape(query_sc, (1, -1))

    # calculate L1 distance
    l1_dist = calculate_l1_distance(query_sc, sf)
    # calculate L2 distance
    l2_dist = calculate_l2_distance(query_sc, sf)

    # assign dummy values if not found
    num_k_sf_nh = -9999
    num_k_sf_nmotb = -9999
    maha_pos_dist = -9999
    maha_neg_dist = -9999


    # start = time.time()
    # get the distances and indices for all the training data (ascending order of distance)
    distances, indices = nbrs.kneighbors(query_sc)
    # print(indices)
    # get the prediction based on top k neighbors (k=3)
    labels = [y_train[index] for index in indices[0][:k_neighbors]]
    pred = max(set(labels), key=labels.count)
    # print(time.time() - start)

    # check if predicted value is same as the actual value
    if pred == y_test[0]:
        # total_correct_pred += 1
        # start = time.time()
        # get nearest hit idx of the query
        nearest_hit_idx = get_nearest_hit_index(pred, indices, y_train)
        # get nmotb idx of the query
        nmotb_idx = get_nmotb_index(pred, indices, distances, nearest_hit_idx, y_train)
        # print(time.time() - start)
        # check if both nearest hit and nmotb exists
        if nearest_hit_idx is not None and nmotb_idx is not None:
            # total_nh_nm += 1
            # get nearest hit and nmotb
            #nearest_hit = x_train[nearest_hit_idx]
            #nmotb = x_train[nmotb_idx]

            # start = time.time()
            train = np.column_stack([x_train, y_train])
            # get positive class from training set
            if pred == 0:
                pos_fltr = 0
                neg_fltr = 1
            elif pred == 1:
                pos_fltr = 1
                neg_fltr = 0
            pos = np.asarray([pos_fltr])
            pos_class = train[np.in1d(train[:, -1], pos)]
            pos_class = pos_class[:, :-1]
            # get negative class from training set
            neg = np.asarray([neg_fltr])
            neg_class = train[np.in1d(train[:, -1], neg)]
            neg_class = neg_class[:, :-1]
            # print(time.time() - start)

            # start = time.time()
            # get mahalanobis distance for positive and negative class
            maha_pos_dist = calculate_mahalanobis_dist(sf, pos_class.astype(float))
            maha_neg_dist = calculate_mahalanobis_dist(sf, neg_class.astype(float))
            # print(time.time() - start)

            # get k values
            # start = time.time()
            num_k_sf_nh, num_k_sf_nmotb = calculate_k(nearest_hit_idx, nmotb_idx, nbrs, sf)
            # print(time.time() - start)
            # calculate trust score

    # print('-------------------------------')

    return sf_val, feat_list, num_k_sf_nh, num_k_sf_nmotb, l1_dist, l2_dist, maha_pos_dist, maha_neg_dist


# divide the inputs into chunks
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


if __name__ == '__main__':

    dataset = pd.read_csv('../datasets/default_credit_card.csv')

    key_feature = 'PAY_AMT3'
    mdn_type = 'higher'  # 'higher' or 'lower'

    target = 'Default'
    distance_metric = 'm_estimate'  # m_estimate or abdm or mvdm

    # select numerical and categorical data
    dataset_sub = dataset.loc[:, dataset.columns != target]

    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)

    num_cols = numerical_columns_selector(dataset_sub)
    cat_cols = categorical_columns_selector(dataset_sub)

    num_idx = [dataset.columns.get_loc(col) for col in num_cols]
    cat_idx = [dataset.columns.get_loc(col) for col in cat_cols]

    feat_order = []
    for feat in dataset.columns:
        feat_order.append(feat)
    feat_order = np.array(feat_order)

    col_types = dataset.dtypes

    # get standard deviation of the numerical features
    std_dict = get_numerical_std(dataset, target)

    # get numerical embeddings of categorical variables
    cat_embed = cat_num_embed(dataset, target, distance_metric)

    # embed categorical columns to numerical values
    dataset_cp = dataset.copy()
    dataset_embed = map_cat_to_num(dataset_cp, target, distance_metric)

    y = np.array(dataset[target])
    x = np.array(dataset_sub)

    key_feat_idx = dataset.columns.get_loc(key_feature)
    # check if the feature is continuous or categorical
    if key_feat_idx in num_idx:
        key_feature_type = 'continuous'
    else:
        key_feature_type = 'categorical'

    # initialize other params
    num_classes = 2
    k_neighbors = 3

    # initialize leave-one-out-corss-validation
    loo = LeaveOneOut()

    # transformer for numerical values
    transformer = MinMaxScaler()

    # for testing
#     x = x[:500]
#     y = y[:500]
#     n = 100

    inputs = [(
              x, y, train_index, test_index, dataset_embed, cat_embed, feat_order, cat_cols, cat_idx, num_cols, num_idx,
              num_classes, target, col_types, distance_metric, key_feature, key_feature_type, mdn_type, k_neighbors,
              std_dict, transformer, key_feat_idx)
              for train_index, test_index in loo.split(x)]

    # number of elements to have in each chunk
    n = 1000

    x = list(divide_chunks(inputs, n))

    final = []

    for val in tqdm(x, total=len(x)):
        with Pool() as p:
            results = p.starmap(sf_loop, val)
            #results = p.starmap(sf_loop, tqdm(inputs, total=len(inputs)))

        p.close()
        p.join()

        final.append(results)


    # write the result dictionary to pickle file
    with open('./results/'+key_feature+'_'+mdn_type+'_mdn.pickle', 'wb') as f:
        pickle.dump(final, f)










