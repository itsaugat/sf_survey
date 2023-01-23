import pandas as pd
import numpy as np
from sklearn.utils import check_random_state
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_selector as selector
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from ../utils/cat_to_num import map_cat_to_num, cat_num_embed
from ../utils/most_distant_neighbor import get_most_distant_neighbor, get_most_distant_neighbor_val
from ../utils/helpers import get_numerical_std, compute_sf_value_continuous, compute_sf_value_categorical, get_nearest_hit_index, get_nmotb_index, calculate_k
from ../utils/metrics import calculate_mahalanobis_dist
from ../utils/kleor_helper import count_num_com, count_cat_com, calculate_l1_distance, calculate_l2_distance
from multiprocessing import Pool
import pickle
import warnings
warnings.simplefilter('ignore', UserWarning)



def loocv_loop(x, y, train_index, test_index, cat_embed, feat_order,
               k_neighbors, cat_idx, num_idx, num_cols, std_dict, transformer, min_num_class, dataset_embed, target):
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
    df[target] = y_train
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
    nugent_dict = {}

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

            # get nugent based semi-factuals
            
            # create local case base
            num_class_0 = 0
            num_class_1 = 0
            last_sim = None
            local_case_x = []
            local_case_y = []
            
            indices = indices[0]
            distances = distances[0]
            
            # iterate through the neighbors
            for i, (idx, dist) in enumerate(zip(indices, distances)):
                # get the class of the neighbor
                idx_class = y_train[idx]

                if (num_class_0 > min_num_class) and (num_class_1 > min_num_class) and (dist != last_sim):
                    break
                else:
                    local_case_x.append(x_train[idx])
                    local_case_y.append(idx_class)

                    if idx_class == 0:
                        num_class_0 += 1
                    elif idx_class == 1:
                        num_class_1 += 1

                    last_sim = dist
            
            # fit logistic regression model on local case base
            lr = LogisticRegression(random_state=0).fit(local_case_x, local_case_y)
            
            # get the probability based on logistic regression for being in the same predicted class
            query_prob = lr.predict_proba(query)[0][list(lr.classes_).index(pred)]
            
            '''filter only query whose probability (predicted by logistic regression model) 
            of being in the actual class is >=0.5'''
            if query_prob >= 0.5:

                '''
                Each case in the local case base (alternatively, the k-neighbors) is considered as the candidate explanation case.
                Get the case which has the least probability of being in the same class as the query. 
                '''
                prob_list = []
                for idx in range(len(local_case_x)):
                    label = local_case_y[idx]
                    probabs = lr.predict_proba(np.reshape(local_case_x[idx], (1, -1)))
                    class_prob = probabs[0][list(lr.classes_).index(pred)]
                    if class_prob >= 0.5 and label==pred:
                        prob_list.append((idx, label, class_prob))

                # sort in ascending order of probability (lowest to highest)
                sorted_prob_list = sorted(prob_list, key=lambda t: t[2])
                
                # select the first instance as the semi-factual
                sf = local_case_x[sorted_prob_list[0][0]]
                
                # compute metrics
                if sf is not None:
                    sf_val_dict = {}
                    feat_diff_dict = {}

                    for feature in feat_order[:-1]:

                        key_feat_idx = df.columns.get_loc(feature)

                        cat_count = count_cat_com(cat_idx, sf, query[0], key_feat_idx)
                        num_count = count_num_com(num_idx, sf, query[0], key_feat_idx, std_dict)

                        if sf[key_feat_idx] > query[0][key_feat_idx]:
                            mdn_type = 'higher'
                            # get highest mdn
                            # convert into dictionary
                            row = {}
                            for a, b in zip(feat_order[:-1], query[0]):
                                row[a] = b

                            # add target column to query at the end
                            row[target] = y_test[0]

                            mdn_val = get_most_distant_neighbor_val(row, target, df, feature, mdn_type)

                            feat_diff = abs(sf[key_feat_idx] - query[0][key_feat_idx])
                            max_feat_diff = abs(mdn_val - query[0][key_feat_idx])

                            sf_val = (num_count + cat_count) / (len(cat_idx) + len(num_idx)) + (
                                        feat_diff / max_feat_diff)

                        elif sf[key_feat_idx] < query[0][key_feat_idx]:
                            mdn_type = 'lower'
                            # get highest mdn
                            # convert into dictionary
                            row = {}
                            for a, b in zip(feat_order[:-1], query[0]):
                                row[a] = b

                            # add target column to query at the end
                            row[target] = y_test[0]

                            mdn_val = get_most_distant_neighbor_val(row, target, df, feature, mdn_type)

                            feat_diff = abs(sf[key_feat_idx] - query[0][key_feat_idx])
                            max_feat_diff = abs(mdn_val - query[0][key_feat_idx])

                            sf_val = (num_count + cat_count) / (len(cat_idx) + len(num_idx)) + (
                                        feat_diff / max_feat_diff)

                        elif sf[key_feat_idx] == query[0][key_feat_idx]:
                            sf_val = (num_count + cat_count) / (len(cat_idx) + len(num_idx))

                        sf_val_dict[feature] = sf_val
                        feat_diff_dict[feature] = (len(cat_idx) + len(num_idx)) - (num_count + cat_count)

                    # get sf_val score
                    key_feature = max(sf_val_dict, key=sf_val_dict.get)
                    sf_val = sf_val_dict[key_feature]

                    feat_diff = feat_diff_dict[key_feature]

                    # calculate L1 distance
                    l1_dist = calculate_l1_distance(query[0], sf)
                    # calculate L2 distance
                    l2_dist = calculate_l2_distance(query[0], sf)

                    # get mahalanobis distance for positive and negative class
                    maha_pos = calculate_mahalanobis_dist(sf, pos_class.astype(float))
                    maha_neg = calculate_mahalanobis_dist(sf, neg_class.astype(float))

                    # get k values
                    num_k_sf_nh, num_k_sf_nmotb = calculate_k(nearest_hit_idx, nmotb_idx, nbrs, sf)
                    
                    # store results to dictionary
                    nugent_dict['key_feature'] = key_feature
                    nugent_dict['sf_val'] = sf_val
                    nugent_dict['feat_diff'] = feat_diff
                    nugent_dict['k_nh'] = num_k_sf_nh
                    nugent_dict['k_nmotb'] = num_k_sf_nmotb
                    nugent_dict['l1_dist'] = l1_dist
                    nugent_dict['l2_dist'] = l2_dist
                    nugent_dict['maha_pos_dist'] = maha_pos
                    nugent_dict['maha_neg_dist'] = maha_neg


    return {'nugent_dict' : nugent_dict}


# divide the inputs into chunks
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


if __name__ == '__main__':

    dt_name = 'diabetes'

    dataset = pd.read_csv('.../datasets/processed/'+dt_name+'.csv')
    #dataset = pd.read_csv('/home/saugat/research/datasets/'+dt_name+'.csv')

    target = 'Outcome'
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
    
    # minimum number of instances for each class in the local case base
    min_num_class = 200

    # for testing
    #x = x[:500]
    #y = y[:500]
    #n = 100

    inputs = [
        (x, y, train_index, test_index, cat_embed, feat_order, k_neighbors, cat_idx, num_idx, num_cols, std_dict, transformer, min_num_class, dataset_embed, target)
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
    with open('./results/nugent_'+dt_name+'.pickle', 'wb') as f:
        pickle.dump(final, f)


