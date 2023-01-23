import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_selector as selector
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from ../utils/cat_to_num import map_cat_to_num, cat_num_embed
from ../utils/most_distant_neighbor import get_most_distant_neighbor, get_most_distant_neighbor_val
from ../utils/helpers import get_numerical_std, compute_sf_value_continuous, compute_sf_value_categorical, get_nearest_hit_index, get_nmotb_index, calculate_k
from ../utils/metrics import calculate_mahalanobis_dist
from ../utils/kleor_helper import get_kleor_sim_miss, get_kleor_global_sim, get_attribute_count, get_kleor_attr_sim, count_num_com, count_cat_com, calculate_l1_distance, calculate_l2_distance
from multiprocessing import Pool
import pickle
import warnings
warnings.simplefilter('ignore', UserWarning)



def loocv_loop(x, y, train_index, test_index, cat_embed, feat_order,
               k_neighbors, cat_idx, num_idx, num_cols, std_dict, transformer, dataset_embed, target):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    ############## if x_train is not already categorically and numerically scaled ################
    # normalize categorical feature values
    for x in x_train:
        for idx in cat_idx:
            x[idx] = float(cat_embed[idx][x[idx]])

    # convert numpy array to df for normalizing numerical features
    df = pd.DataFrame(x_train, columns=feat_order[:-1])
    # normalize numerical feature values
    df[num_cols] = transformer.fit_transform(df[num_cols])
    # convert df to numpy array
    x_train = df.to_numpy()

    # fit nearest neighbor on x_train
    num_neighbors = len(x_train)
    nbrs = NearestNeighbors(n_neighbors=num_neighbors).fit(x_train)

    query = x_test[0]

    # if query is not scaled
    for idx in cat_idx:
        query[idx] = float(cat_embed[idx][query[idx]])

    query[num_idx] = transformer.transform(np.reshape(query[num_idx], (1, -1)))

    # reshape query instance
    query = np.reshape(query, (1, -1))
    # get the distances and indices for all the training data (ascending order of distance)
    distances, indices = nbrs.kneighbors(query)

    labels = [y_train[index] for index in indices[0][:k_neighbors]]
    pred = max(set(labels), key=labels.count)

    # create dictionaries to hold results
    sim_miss_dict = {}
    global_sim_dict = {}
    attr_sim_dict = {}

    if pred == y_test[0]:

        # get nearest hit idx of the query
        nearest_hit_idx = get_nearest_hit_index(pred, indices, y_train)
        # print(nearest_hit_idx)
        # get nmotb idx of the query
        nmotb_idx = get_nmotb_index(pred, indices, distances, nearest_hit_idx, y_train)
        # print(nmotb_idx)

        if nearest_hit_idx is not None and nmotb_idx is not None:

            # get nearest hit and nmotb
            nearest_hit = x_train[nearest_hit_idx]
            nmotb = x_train[nmotb_idx]

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

            # get kleor based semi-factuals
            kleor_sim_miss = get_kleor_sim_miss(pred, train, nmotb)
            kleor_global_sim = get_kleor_global_sim(pred, train, nmotb, query)
            kleor_attr_sim = get_kleor_attr_sim(pred, train, nmotb, query, num_idx, cat_idx)


            # compute metrics for kleor_sim_miss
            if kleor_sim_miss is not None:

                sf_val_dict = {}
                feat_diff_dict = {}

                for feature in feat_order[:-1]:

                    key_feat_idx = df.columns.get_loc(feature)

                    cat_count = count_cat_com(cat_idx, kleor_sim_miss, query[0], key_feat_idx)
                    num_count = count_num_com(num_idx, kleor_sim_miss, query[0], key_feat_idx, std_dict)

                    if kleor_sim_miss[key_feat_idx] > query[0][key_feat_idx]:
                        mdn_type = 'higher'
                        # get highest mdn
                        # convert into dictionary
                        row = {}
                        for a, b in zip(feat_order[:-1], query[0]):
                            row[a] = b

                        # add target column to query at the end
                        row[target] = y_test[0]

                        mdn_val = get_most_distant_neighbor_val(row, target, dataset_embed, feature, mdn_type)

                        feat_diff = abs(kleor_sim_miss[key_feat_idx] - query[0][key_feat_idx])
                        max_feat_diff = abs(mdn_val - query[0][key_feat_idx])

                        sf_val = (num_count + cat_count) / (len(cat_idx) + len(num_idx)) + (feat_diff / max_feat_diff)

                    elif kleor_sim_miss[key_feat_idx] < query[0][key_feat_idx]:
                        mdn_type = 'lower'
                        # get highest mdn
                        # convert into dictionary
                        row = {}
                        for a, b in zip(feat_order[:-1], query[0]):
                            row[a] = b

                        # add target column to query at the end
                        row[target] = y_test[0]

                        mdn_val = get_most_distant_neighbor_val(row, target, dataset_embed, feature, mdn_type)

                        feat_diff = abs(kleor_sim_miss[key_feat_idx] - query[0][key_feat_idx])
                        max_feat_diff = abs(mdn_val - query[0][key_feat_idx])

                        sf_val = (num_count + cat_count) / (len(cat_idx) + len(num_idx)) + (feat_diff / max_feat_diff)

                    elif kleor_sim_miss[key_feat_idx] == query[0][key_feat_idx]:
                        sf_val = (num_count + cat_count) / (len(cat_idx) + len(num_idx))

                    sf_val_dict[feature] = sf_val
                    feat_diff_dict[feature] = (len(cat_idx) + len(num_idx)) - (num_count + cat_count)

                # get sf_val score
                key_feature = max(sf_val_dict, key=sf_val_dict.get)
                sim_miss_sf_val = sf_val_dict[key_feature]

                # get num of feature diff between query and kleor_sim_miss
                # cat_count = count_cat_com(cat_idx, kleor_sim_miss, query[0])
                # num_count = count_num_com(num_idx, kleor_sim_miss, query[0], std_dict)
                # sim_miss_feat_diff = (len(cat_idx) + len(num_idx)) - (num_count + cat_count)
                sim_miss_feat_diff = feat_diff_dict[key_feature]


                # calculate L1 distance
                sim_miss_l1 = calculate_l1_distance(query[0], kleor_sim_miss)
                # calculate L2 distance
                sim_miss_l2 = calculate_l2_distance(query[0], kleor_sim_miss)

                # get mahalanobis distance for positive and negative class
                sim_miss_maha_pos = calculate_mahalanobis_dist(kleor_sim_miss, pos_class.astype(float))
                sim_miss_maha_neg = calculate_mahalanobis_dist(kleor_sim_miss, neg_class.astype(float))

                # get k values
                sim_miss_num_k_sf_nh, sim_miss_num_k_sf_nmotb = calculate_k(nearest_hit_idx, nmotb_idx, nbrs,
                                                                            kleor_sim_miss)

                # store results to dictionary
                sim_miss_dict['key_feature'] = key_feature
                sim_miss_dict['sf_val'] = sim_miss_sf_val
                sim_miss_dict['feat_diff'] = sim_miss_feat_diff
                sim_miss_dict['k_nh'] = sim_miss_num_k_sf_nh
                sim_miss_dict['k_nmotb'] = sim_miss_num_k_sf_nmotb
                sim_miss_dict['l1_dist'] = sim_miss_l1
                sim_miss_dict['l2_dist'] = sim_miss_l2
                sim_miss_dict['maha_pos_dist'] = sim_miss_maha_pos
                sim_miss_dict['maha_neg_dist'] = sim_miss_maha_neg

            # compute metrics for kleor_global_sim
            if kleor_global_sim is not None:

                sf_val_dict = {}
                feat_diff_dict = {}

                for feature in feat_order[:-1]:

                    key_feat_idx = df.columns.get_loc(feature)

                    cat_count = count_cat_com(cat_idx, kleor_global_sim, query[0], key_feat_idx)
                    num_count = count_num_com(num_idx, kleor_global_sim, query[0], key_feat_idx, std_dict)

                    if kleor_global_sim[key_feat_idx] > query[0][key_feat_idx]:
                        mdn_type = 'higher'
                        # get highest mdn
                        # convert into dictionary
                        row = {}
                        for a, b in zip(feat_order[:-1], query[0]):
                            row[a] = b

                        # add target column to query at the end
                        row[target] = y_test[0]

                        mdn_val = get_most_distant_neighbor_val(row, target, dataset_embed, feature, mdn_type)

                        feat_diff = abs(kleor_global_sim[key_feat_idx] - query[0][key_feat_idx])
                        max_feat_diff = abs(mdn_val - query[0][key_feat_idx])

                        sf_val = (num_count + cat_count) / (len(cat_idx) + len(num_idx)) + (feat_diff / max_feat_diff)

                    elif kleor_global_sim[key_feat_idx] < query[0][key_feat_idx]:
                        mdn_type = 'lower'
                        # get highest mdn
                        # convert into dictionary
                        row = {}
                        for a, b in zip(feat_order[:-1], query[0]):
                            row[a] = b

                        # add target column to query at the end
                        row[target] = y_test[0]

                        mdn_val = get_most_distant_neighbor_val(row, target, dataset_embed, feature, mdn_type)

                        feat_diff = abs(kleor_global_sim[key_feat_idx] - query[0][key_feat_idx])
                        max_feat_diff = abs(mdn_val - query[0][key_feat_idx])

                        sf_val = (num_count + cat_count) / (len(cat_idx) + len(num_idx)) + (feat_diff / max_feat_diff)

                    elif kleor_global_sim[key_feat_idx] == query[0][key_feat_idx]:
                        sf_val = (num_count + cat_count) / (len(cat_idx) + len(num_idx))

                    sf_val_dict[feature] = sf_val
                    feat_diff_dict[feature] = (len(cat_idx) + len(num_idx)) - (num_count + cat_count)

                # get sf_val score
                key_feature = max(sf_val_dict, key=sf_val_dict.get)
                global_sim_sf_val = sf_val_dict[key_feature]

                # get num of feature diff between query and kleor_sim_miss
                global_sim_feat_diff = feat_diff_dict[key_feature]

                # calculate L1 distance
                global_sim_l1 = calculate_l1_distance(query[0], kleor_global_sim)
                # calculate L2 distance
                global_sim_l2 = calculate_l2_distance(query[0], kleor_global_sim)

                # get mahalanobis distance for positive and negative class
                global_sim_maha_pos = calculate_mahalanobis_dist(kleor_global_sim, pos_class.astype(float))
                global_sim_maha_neg = calculate_mahalanobis_dist(kleor_global_sim, neg_class.astype(float))

                # get k values
                global_sim_num_k_sf_nh, global_sim_num_k_sf_nmotb = calculate_k(nearest_hit_idx, nmotb_idx, nbrs,
                                                                                kleor_global_sim)

                # store results to dictionary
                global_sim_dict['key_feature'] = key_feature
                global_sim_dict['sf_val'] = global_sim_sf_val
                global_sim_dict['feat_diff'] = global_sim_feat_diff
                global_sim_dict['k_nh'] = global_sim_num_k_sf_nh
                global_sim_dict['k_nmotb'] = global_sim_num_k_sf_nmotb
                global_sim_dict['l1_dist'] = global_sim_l1
                global_sim_dict['l2_dist'] = global_sim_l2
                global_sim_dict['maha_pos_dist'] = global_sim_maha_pos
                global_sim_dict['maha_neg_dist'] = global_sim_maha_neg

            # compute metrics for kleor_attr_sim
            if kleor_attr_sim is not None:

                sf_val_dict = {}
                feat_diff_dict = {}

                for feature in feat_order[:-1]:

                    key_feat_idx = df.columns.get_loc(feature)

                    cat_count = count_cat_com(cat_idx, kleor_attr_sim, query[0], key_feat_idx)
                    num_count = count_num_com(num_idx, kleor_attr_sim, query[0], key_feat_idx, std_dict)

                    if kleor_attr_sim[key_feat_idx] > query[0][key_feat_idx]:
                        mdn_type = 'higher'
                        # get highest mdn
                        # convert into dictionary
                        row = {}
                        for a, b in zip(feat_order[:-1], query[0]):
                            row[a] = b

                        # add target column to query at the end
                        row[target] = y_test[0]

                        mdn_val = get_most_distant_neighbor_val(row, target, dataset_embed, feature, mdn_type)

                        feat_diff = abs(kleor_attr_sim[key_feat_idx] - query[0][key_feat_idx])
                        max_feat_diff = abs(mdn_val - query[0][key_feat_idx])

                        sf_val = (num_count + cat_count) / (len(cat_idx) + len(num_idx)) + (feat_diff / max_feat_diff)

                    elif kleor_attr_sim[key_feat_idx] < query[0][key_feat_idx]:
                        mdn_type = 'lower'
                        # get highest mdn
                        # convert into dictionary
                        row = {}
                        for a, b in zip(feat_order[:-1], query[0]):
                            row[a] = b

                        # add target column to query at the end
                        row[target] = y_test[0]

                        mdn_val = get_most_distant_neighbor_val(row, target, dataset_embed, feature, mdn_type)

                        feat_diff = abs(kleor_attr_sim[key_feat_idx] - query[0][key_feat_idx])
                        max_feat_diff = abs(mdn_val - query[0][key_feat_idx])

                        sf_val = (num_count + cat_count) / (len(cat_idx) + len(num_idx)) + (feat_diff / max_feat_diff)

                    elif kleor_attr_sim[key_feat_idx] == query[0][key_feat_idx]:
                        sf_val = (num_count + cat_count) / (len(cat_idx) + len(num_idx))

                    sf_val_dict[feature] = sf_val
                    feat_diff_dict[feature] = (len(cat_idx) + len(num_idx)) - (num_count + cat_count)

                # get sf_val score
                key_feature = max(sf_val_dict, key=sf_val_dict.get)
                attr_sim_sf_val = sf_val_dict[key_feature]

                # get num of feature diff between query and kleor_sim_miss
                attr_sim_feat_diff = feat_diff_dict[key_feature]

                # calculate L1 distance
                attr_sim_l1 = calculate_l1_distance(query[0], kleor_attr_sim)
                # calculate L2 distance
                attr_sim_l2 = calculate_l2_distance(query[0], kleor_attr_sim)

                # get mahalanobis distance for positive and negative class
                attr_sim_maha_pos = calculate_mahalanobis_dist(kleor_attr_sim, pos_class.astype(float))
                attr_sim_maha_neg = calculate_mahalanobis_dist(kleor_attr_sim, neg_class.astype(float))

                # get k values
                attr_sim_num_k_sf_nh, attr_sim_num_k_sf_nmotb = calculate_k(nearest_hit_idx, nmotb_idx, nbrs,
                                                                            kleor_attr_sim)

                # store results to dictionary
                attr_sim_dict['key_feature'] = key_feature
                attr_sim_dict['sf_val'] = attr_sim_sf_val
                attr_sim_dict['feat_diff'] = attr_sim_feat_diff
                attr_sim_dict['k_nh'] = attr_sim_num_k_sf_nh
                attr_sim_dict['k_nmotb'] = attr_sim_num_k_sf_nmotb
                attr_sim_dict['l1_dist'] = attr_sim_l1
                attr_sim_dict['l2_dist'] = attr_sim_l2
                attr_sim_dict['maha_pos_dist'] = attr_sim_maha_pos
                attr_sim_dict['maha_neg_dist'] = attr_sim_maha_neg


    return {'sim_miss_dict' : sim_miss_dict,
            'global_sim_dict' : global_sim_dict,
            'attr_sim_dict' : attr_sim_dict}


# divide the inputs into chunks
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


if __name__ == '__main__':

    dt_name = 'default_credit_card'

    dataset = pd.read_csv('.../datasets/processed/'+dt_name+'.csv')
    #dataset = pd.read_csv('/home/saugat/research/datasets/'+dt_name+'.csv')

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

    # get standard deviation of the numerical features
    std_dict = get_numerical_std(dataset, target)

    # get numerical embeddings of categorical variables
    cat_embed = cat_num_embed(dataset, target, distance_metric)

    # embed categorical columns to numerical values
    dataset_cp = dataset.copy()
    dataset_embed = map_cat_to_num(dataset_cp, target, distance_metric)

    y = np.array(dataset[target])
    x = np.array(dataset_sub)

    k_neighbors = 3

    # initialize leave-one-out-cross-validation
    loo = LeaveOneOut()

    # transformer for numerical values
    transformer = MinMaxScaler()

    # for testing
    # x = x[:500]
    # y = y[:500]
    # n = 100

    inputs = [
        (x, y, train_index, test_index, cat_embed, feat_order, k_neighbors, cat_idx, num_idx, num_cols, std_dict, transformer, dataset_embed, target)
        for train_index, test_index in loo.split(x)]

    # number of elements to have in each chunk
    n = 1000

    x = list(divide_chunks(inputs, n))

    final = []

    for val in tqdm(x, total=len(x)):
        with Pool() as p:
            results = p.starmap(loocv_loop, val)
            # results = p.starmap(sf_loop, tqdm(inputs, total=len(inputs)))

        p.close()
        p.join()

        final.append(results)

    #write the result dictionary to pickle file
    with open('./results/kleor_'+dt_name+'.pickle', 'wb') as f:
        pickle.dump(final, f)


