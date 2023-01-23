import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from scipy.spatial import distance
#import seaborn as sns
#import matplotlib.pyplot as plt
#from IPython.display import set_matplotlib_formats
from cat_to_num import map_cat_to_num, cat_num_embed



'''
compute distance measure between two vectors
'''
def compute_distance (v1, v2):
    return distance.euclidean(v1, v2)


'''
get the categorical feature based on similar or closest value
'''
def get_categorical(val, cat_list):
    diff = lambda list_value : abs(list_value - val)
    closest_value = min(cat_list, key=diff)
    return closest_value


'''
get higher set for a given query
'''
def get_higher_set(query, target_col, dataset_embed, categorical_columns, categoric_attr_idx, cat_embed, feature):
    
    # get query label
    query_label = query[target_col]
    
    # remove target key from query
    query.popitem()
            
    # filter dataset based on the label of the query
    df_filter = dataset_embed[dataset_embed[target_col] == query_label]
    df_filter = df_filter.drop(columns=[target_col], axis=1)
    
    # index of the key feature
    num_idx = df_filter.columns.get_loc(feature)

    # sort values in ascending order
    filter_df = df_filter.sort_values(by=[feature])
    
    # convert df to dict to iterate faster
    filter_df_dict = filter_df.to_dict('records')

    # normalize categorical feature values in query
    for cat_feature in categorical_columns:
        cat_idx = df_filter.columns.get_loc(cat_feature)
        query[cat_feature] = float(cat_embed[cat_idx][query[cat_feature]])
    
    
    # find higher_set and lower_set
    row_idx = 0
    for row in filter_df_dict:
        if row[feature] >= query[feature]:
            threshold = row_idx
            break
        row_idx += 1
    
    higher_set = filter_df[threshold:].values.tolist()
    
    # get the categorical variables from the numerical embeddings
    for item in higher_set:
        for idx in categoric_attr_idx:
            val = item[idx]
            cat_list = list(cat_embed[idx].values())
            # get the index of the closest categorical value
            cat_idx = list(cat_embed[idx].values()).index(get_categorical(val, cat_list))
            item[idx] = list(cat_embed[idx])[cat_idx]
    
    return higher_set


'''
get lower set for a given query
'''
def get_lower_set(query, target_col, dataset_embed, categorical_columns, categoric_attr_idx, cat_embed, feature):
    
    # get query label
    query_label = query[target_col]
    
    # remove target key from query
    query.popitem()
            
    # filter dataset based on the label of the query
    df_filter = dataset_embed[dataset_embed[target_col] == query_label]
    df_filter = df_filter.drop(columns=[target_col], axis=1)
    
    # index of the key feature
    num_idx = df_filter.columns.get_loc(feature)

    # sort values in ascending order
    filter_df = df_filter.sort_values(by=[feature])
    
    # convert df to dict to iterate faster
    filter_df_dict = filter_df.to_dict('records')

    # normalize categorical feature values in query
    for cat_feature in categorical_columns:
        cat_idx = df_filter.columns.get_loc(cat_feature)
        query[cat_feature] = float(cat_embed[cat_idx][query[cat_feature]])
    
    
    # find higher_set and lower_set
    row_idx = 0
    for row in filter_df_dict:
        if row[feature] >= query[feature]:
            threshold = row_idx
            break
        row_idx += 1
    
    lower_set = filter_df[:threshold].values.tolist()
    
    # get the categorical variables from the numerical embeddings
    for item in lower_set:
        for idx in categoric_attr_idx:
            val = item[idx]
            cat_list = list(cat_embed[idx].values())
            # get the index of the closest categorical value
            cat_idx = list(cat_embed[idx].values()).index(get_categorical(val, cat_list))
            item[idx] = list(cat_embed[idx])[cat_idx]
    
    return lower_set
    

'''
get mdn for a particular feature and type
'''
def get_most_distant_neighbor(query, target_col, dataset_embed, categorical_columns, categoric_attr_idx, numerical_columns, numeric_attr_idx, cat_embed, cat_dist_metric, feature, mdn_type):
    
    #query_temp = query.copy()
    
    # get query label
    query_label = query[target_col]
    
    # remove target key from query
    query.popitem()
            
    # filter dataset based on the label of the query
    df_filter = dataset_embed[dataset_embed[target_col] == query_label]
    df_filter = df_filter.drop(columns=[target_col], axis=1)
            
    # create transformer object
    transformer = MinMaxScaler()

    mdn_list = {}
    
    # index of the key feature
    num_idx = df_filter.columns.get_loc(feature)

    # sort values in ascending order
    filter_df = df_filter.sort_values(by=[feature])

    # convert df to dict to iterate faster
    filter_df_dict = filter_df.to_dict('records')

    # normalize categorical feature values in query
    for cat_feature in categorical_columns:
        cat_idx = df_filter.columns.get_loc(cat_feature)
        query[cat_feature] = float(cat_embed[cat_idx][query[cat_feature]])

    # find higher_set and lower_set
    row_idx = 0
    threshold = None
    for row in filter_df_dict:
        if row[feature] >= query[feature]:
            threshold = row_idx
            break
        row_idx += 1
        
    
    # if threshold is still 'None', take the last index as threshold
    if threshold is None:
        threshold = row_idx


    filter_df_sc = filter_df.copy()

    #filter_df = filter_df.to_numpy()
    #filter_df_sc = np.copy(filter_df)

    # normalize numerical feature values in the dataset
    filter_df_sc[numerical_columns] = transformer.fit_transform(filter_df_sc[numerical_columns])
    #print(filter_df_sc[:2])

    #print(filter_df.head(2))
    #print(filter_df_sc.head(2))

    # normalize the query
    query_np = np.array(list(query.values()))
    # transform numerical feature values
    query_np[numeric_attr_idx] = transformer.transform(np.reshape(query_np[numeric_attr_idx] , (1, -1)))

    # convert to float
    query_np = query_np.astype(float)
    
    if mdn_type == 'higher':
        # inclusive of the threshold index
        higher_set_sc = filter_df_sc[threshold:].to_numpy()
        higher_set = filter_df[threshold:].to_numpy()
        
        high_dict_list = {}
        for i in range(len(higher_set)):
            # create dictionary of feature values     
            high_dict_list[i] = higher_set[i][num_idx]
        
        high_df = pd.DataFrame(high_dict_list.items(), columns=['index', feature])
        
        '''
        compute distance measure from query to each instance in the higher_set
        excluding the feature value in consideration
        '''
        high_dist_list = []
        temp_query = np.delete(query_np.flatten(), num_idx)
        for i in range(len(higher_set_sc)):
            # remove the key feature before computing distance
            temp_arr = np.delete(higher_set_sc[i], num_idx)
            #temp_query = np.delete(query_np.flatten(), num_idx)
            high_dist_list.append(compute_distance(temp_query, temp_arr))
        
        high_df['distance'] = high_dist_list
        
        high_df_sort = high_df.sort_values([feature, 'distance'], ascending = [False, False])
        
        # remove other columns from df
        high_df_sort = high_df_sort.drop([feature, 'distance'], axis=1)
                
        mdn_df_list = high_df_sort.values.tolist()
        
        for item in mdn_df_list:
            #high_df_list.append(higher_set[int(item[0])])
            item[0] = higher_set[int(item[0])].tolist()
                
#         mdn_df_list = mdn_df_list[0][0].tolist()
                
        # get the categorical variables from the numerical embeddings
        for mdn in mdn_df_list:
            for idx in categoric_attr_idx:
                val = mdn[0][idx]
                cat_list = list(cat_embed[idx].values())
                # get the index of the closest categorical value
                cat_idx = list(cat_embed[idx].values()).index(get_categorical(val, cat_list))
                mdn[0][idx] = list(cat_embed[idx])[cat_idx]
            
    
    elif mdn_type == 'lower':
        # ':' is not inclusive of the threshold index, so adding '1'
        lower_set_sc = filter_df_sc[:(threshold+1)].to_numpy()
        lower_set = filter_df[:(threshold+1)].to_numpy()
        
        low_dict_list = {}
        for i in range(len(lower_set)):
            # create dictionary of feature values
            low_dict_list[i] = lower_set[i][num_idx]
        
        low_df = pd.DataFrame(low_dict_list.items(), columns=['index', feature])
        
        # compute distance measure (euclidean) from query to each instance in the higher_set
        low_dist_list = []
        temp_query = np.delete(query_np.flatten(), num_idx)
        for i in range(len(lower_set_sc)):
            # remove the key feature before computing distance
            temp_arr = np.delete(lower_set_sc[i], num_idx)
            low_dist_list.append(compute_distance(temp_query, temp_arr))
        
        low_df['distance'] = low_dist_list
        
        low_df_sort = low_df.sort_values([feature, 'distance'], ascending = [True, False])
        
        # remove other columns from df
        low_df_sort = low_df_sort.drop([feature, 'distance'], axis=1)
                
        mdn_df_list = low_df_sort.values.tolist()
        
        for item in mdn_df_list:
            #low_df_list.append(lower_set[int(item[0])])
            item[0] = lower_set[int(item[0])].tolist()
        
        #mdn_df_list = mdn_df_list[0][0].tolist()
                
        # get the categorical variables from the numerical embeddings
        for mdn in mdn_df_list:
            for idx in categoric_attr_idx:
                val = mdn[0][idx]
                cat_list = list(cat_embed[idx].values())
                # get the index of the closest categorical value
                cat_idx = list(cat_embed[idx].values()).index(get_categorical(val, cat_list))
                mdn[0][idx] = list(cat_embed[idx])[cat_idx]
        
    return mdn_df_list


