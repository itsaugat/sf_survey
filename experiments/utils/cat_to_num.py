import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple, Union, Callable, List
#from discretizer import Discretizer
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import MDS
from sklearn.compose import make_column_selector as selector
import category_encoders as ce


class Discretizer(object):

    def __init__(self, data: np.ndarray, categorical_features: List[int], feature_names: List[str],
                 percentiles: List[int] = [25, 50, 75]) -> None:
        """
        Initialize the discretizer.
        Parameters
        ----------
        data
            Data to discretize
        categorical_features
            List of indices corresponding to the categorical columns. These features will not be discretized.
            The other features will be considered continuous and therefore discretized.
        feature_names
            List with feature names
        percentiles
            Percentiles used for discretization
        """
        self.to_discretize = ([x for x in range(data.shape[1]) if x not in categorical_features])
        self.percentiles = percentiles

        bins = self.bins(data)
        bins = [np.unique(x) for x in bins]

        self.names = {}  # type: Dict[int, list]
        self.lambdas = {}  # type: Dict[int, Callable]
        for feature, qts in zip(self.to_discretize, bins):

            # get nb of borders (nb of bins - 1) and the feature name
            n_bins = qts.shape[0]
            name = feature_names[feature]

            # create names for bins of discretized features
            self.names[feature] = ['%s <= %.2f' % (name, qts[0])]
            for i in range(n_bins - 1):
                self.names[feature].append('%.2f < %s <= %.2f' % (qts[i], name, qts[i + 1]))
            self.names[feature].append('%s > %.2f' % (name, qts[n_bins - 1]))
            self.lambdas[feature] = lambda x, qts = qts: np.searchsorted(qts, x)

    def bins(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Parameters
        ----------
        data
            Data to discretize
        Returns
        -------
        List with bin values for each feature that is discretized.
        """
        bins = []
        for feature in self.to_discretize:
            qts = np.array(np.percentile(data[:, feature], self.percentiles))
            bins.append(qts)
        return bins

    def discretize(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        data
            Data to discretize
        Returns
        -------
        Discretized version of data with the same dimension.
        """
        data_disc = data.copy()
        for feature in self.lambdas:
            if len(data.shape) == 1:
                data_disc[feature] = int(self.lambdas[feature](data_disc[feature]))
            else:
                data_disc[:, feature] = self.lambdas[feature](data_disc[:, feature]).astype(int)
        return data_disc



'''Depends upond how the data is pre-processed '''

def abdm_dist(X: np.ndarray, cat_vars: dict, cat_vars_bin: dict = dict()) -> dict:
    """
    Calculate the pair-wise distances between categories of a categorical variable using
    the Association-Based Distance Metric based on Le et al (2005).
    http://www.jaist.ac.jp/~bao/papers/N26.pdf
    Parameters
    ----------
    X
        Batch of arrays.
    cat_vars
        Dict with as keys the categorical columns and as optional values
        the number of categories per categorical variable.
    cat_vars_bin
        Dict with as keys the binned numerical columns and as optional values
        the number of bins per variable.
    Returns
    -------
    Dict with as keys the categorical columns and as values the pairwise distance matrix for the variable.
    """
    # TODO: handle triangular inequality
    # ensure numerical stability
    eps = 1e-12

    # infer number of categories per categorical variable
    cat_cols = list(cat_vars.keys())
    for col in cat_cols:
        if cat_vars[col] is not None:
            continue
        cat_vars[col] = len(np.unique(X[:, col]))

    # combine dict for categorical with binned features
    cat_vars_combined = {**cat_vars, **cat_vars_bin}

    d_pair = {}  # type: Dict
    X_cat_eq = {}  # type: Dict
    #print(cat_vars.items())
    for col, n_cat in cat_vars.items():
        #print(col)
        X_cat_eq[col] = []
        for i in range(n_cat):  # for each category in categorical variable, store instances of each category
            #print(i)
            idx = np.where(X[:, col] == i)[0]
            #print(idx)
            X_cat_eq[col].append(X[idx, :])

        #print(X_cat_eq[col])
        # conditional probabilities, also use the binned numerical features
        p_cond = []
        for col_t, n_cat_t in cat_vars_combined.items():
            if col == col_t:
                continue
            p_cond_t = np.zeros([n_cat_t, n_cat])
            for i in range(n_cat_t):
                #print(i)
                for j, X_cat_j in enumerate(X_cat_eq[col]):
                    #print(X_cat_j)
                    idx = np.where(X_cat_j[:, col_t] == i)[0]
                    #print(idx)
                    p_cond_t[i, j] = len(idx) / (X_cat_j.shape[0] + eps)
            p_cond.append(p_cond_t)
        
        #print(p_cond)
        # pairwise distance matrix
        d_pair_col = np.zeros([n_cat, n_cat])
        for i in range(n_cat):
            j = 0
            while j < i:
                d_ij_tmp = 0
                for p in p_cond:  # loop over other categorical variables
                    for t in range(p.shape[0]):  # loop over categories of each categorical variable
                        a, b = p[t, i], p[t, j]
                        d_ij_t = a * np.log((a + eps) / (b + eps)) + b * np.log((b + eps) / (a + eps))  # KL divergence
                        d_ij_tmp += d_ij_t
                d_pair_col[i, j] = d_ij_tmp
                j += 1
        d_pair_col += d_pair_col.T
        d_pair[col] = d_pair_col
    
    
    # get scaled value of each categorical variable
    abdm_abs = {}
    d_min, d_max = 1e10, 0
    for k, v in d_pair.items():
        # distance smoothening
        v **= 1
        # fit multi-dimensional scaler
        mds = MDS(n_components=2, max_iter=5000, eps=1e-9, random_state=0, n_init=4,
                  dissimilarity="precomputed", metric=True)
        d_fit = mds.fit(v)
        emb = d_fit.embedding_  # coordinates in embedding space
        # use biggest single observation Frobenius norm as origin
        origin = np.argsort(np.linalg.norm(emb, axis=1))[-1]
        # calculate distance from origin for each category
        d_origin = np.linalg.norm(emb - emb[origin].reshape(1, -1), axis=1)
        # assign to category
        abdm_abs[k] = d_origin
        d_min_k, d_max_k = d_origin.min(), d_origin.max()
        d_min = d_min_k if d_min_k < d_min else d_min
        d_max = d_max_k if d_max_k > d_max else d_max

    #print(d_max)
    abdm_abs_scaled = {}

    for k, v in abdm_abs.items():
        abdm_scaled = (v - d_min) / (d_max - d_min)
        # center the numerical feature values between the min and max feature range
        abdm_scaled -= .5 * (abdm_scaled.max() + abdm_scaled.min())
        abdm_abs_scaled[k] = np.round(abdm_scaled, 5)  # scaled distance from the origin for each category
        
    return abdm_abs_scaled


def mvdm_dist(X: np.ndarray, y: np.ndarray, cat_vars: dict, alpha: int = 1) -> Dict[Any, np.ndarray]:
    """
    Calculate the pair-wise distances between categories of a categorical variable using
    the Modified Value Difference Measure based on Cost et al (1993).
    https://link.springer.com/article/10.1023/A:1022664626993
    Parameters
    ----------
    X
        Batch of arrays.
    y
        Batch of labels or predictions.
    cat_vars
        Dict with as keys the categorical columns and as optional values
        the number of categories per categorical variable.
    alpha
        Power of absolute difference between conditional probabilities.
    Returns
    -------
    Dict with as keys the categorical columns and as values the pairwise distance matrix for the variable.
    """
    # TODO: handle triangular inequality
    # infer number of categories per categorical variable
    n_y = len(np.unique(y))
    cat_cols = list(cat_vars.keys())
    for col in cat_cols:
        if cat_vars[col] is not None:
            continue
        cat_vars[col] = len(np.unique(X[:, col]))

    # conditional probabilities and pairwise distance matrix
    d_pair = {}
    for col, n_cat in cat_vars.items():
        d_pair_col = np.zeros([n_cat, n_cat])
        p_cond_col = np.zeros([n_cat, n_y])
        for i in range(n_cat):
            idx = np.where(X[:, col] == i)[0]
            for i_y in range(n_y):
                p_cond_col[i, i_y] = np.sum(y[idx] == i_y) / (y[idx].shape[0] + 1e-12)

        for i in range(n_cat):
            j = 0
            while j < i:  # symmetrical matrix
                d_pair_col[i, j] = np.sum(np.abs(p_cond_col[i, :] - p_cond_col[j, :]) ** alpha)
                j += 1
        d_pair_col += d_pair_col.T
        d_pair[col] = d_pair_col
    
    
    # get scaled value of each categorical variable
    mvdm_abs = {}
    d_min, d_max = 1e10, 0
    for k, v in d_pair.items():
        # distance smoothening
        v **= 1
        # fit multi-dimensional scaler
        mds = MDS(n_components=2, max_iter=5000, eps=1e-9, random_state=0, n_init=4,
                  dissimilarity="precomputed", metric=True)
        d_fit = mds.fit(v)
        emb = d_fit.embedding_  # coordinates in embedding space
        # use biggest single observation Frobenius norm as origin
        origin = np.argsort(np.linalg.norm(emb, axis=1))[-1]
        # calculate distance from origin for each category
        d_origin = np.linalg.norm(emb - emb[origin].reshape(1, -1), axis=1)
        # assign to category
        mvdm_abs[k] = d_origin
        d_min_k, d_max_k = d_origin.min(), d_origin.max()
        d_min = d_min_k if d_min_k < d_min else d_min
        d_max = d_max_k if d_max_k > d_max else d_max

    #print(d_max)
    mvdm_abs_scaled = {}

    for k, v in mvdm_abs.items():
        mvdm_scaled = (v - d_min) / (d_max - d_min)
        # center the numerical feature values between the min and max feature range
        mvdm_scaled -= .5 * (mvdm_scaled.max() + mvdm_scaled.min())
        mvdm_abs_scaled[k] = np.round(mvdm_scaled, 5)  # scaled distance from the origin for each category
    
    return mvdm_abs_scaled


def cat_num (row, cat_col_idx, category_map, num_dist):
    for i in range(len(category_map[cat_col_idx])):
        if row[cat_col_idx] == category_map[cat_col_idx][i]:
            return num_dist[cat_col_idx][i]
          
# main function to map categorical values to numerical embedding
def map_cat_to_num (df_orig, target, dist_metric):
    
    df = df_orig.copy()
    Y = df[target]
    Y = np.array(Y)
    df = df.drop(columns=[target], axis=1)
    
    # select numerical and categorical data 
    categorical_columns_selector = selector(dtype_include=object)
    categorical_columns = categorical_columns_selector(df)
    
    if dist_metric == 'm_estimate':
        
        encoder = ce.MEstimateEncoder(cols=categorical_columns)
        
        x = df
        y = Y
        x_tf = encoder.fit_transform(x, y)
        
        # add target column
        x_tf[target] = y.tolist()
        
        return x_tf
   
    else:
    
        # get categorical features and apply labelencoding
        features = list(df.columns)
        category_map = {}
        for f in categorical_columns:
            le = LabelEncoder()
            data_tmp = le.fit_transform(df[f].values)
            df[f] = data_tmp
            category_map[features.index(f)] = list(le.classes_)

        cat_vars = {}

        for f in categorical_columns:
            cat_vars[features.index(f)] = df[f].nunique()


        # convert df to numpy array
        df_np = df.to_numpy()

        disc_perc = [25, 50, 75]

        X_ord, cat_vars_ord = df_np, cat_vars

        # bin numerical features to compute the pairwise distance matrices
        cat_keys = list(cat_vars_ord.keys())
        n_ord = X_ord.shape[1]
        if len(cat_keys) != n_ord:
            fnames = [str(_) for _ in range(n_ord)]
            disc = Discretizer(X_ord, cat_keys, fnames, percentiles=disc_perc)
            X_bin = disc.discretize(X_ord)
            cat_vars_bin = {k: len(disc.names[k]) for k in range(n_ord) if k not in cat_keys}
        else:
            X_bin = X_ord
            cat_vars_bin = {}


        if dist_metric == 'abdm':
            num_dist = abdm_dist(X_bin, cat_vars_ord, cat_vars_bin)
        elif dist_metric == 'mvdm':
            num_dist = mvdm_dist(X_ord, Y, cat_vars_ord, alpha=1)


        # map numerical value of each categorical variable for each categorical column to original df
        for column in categorical_columns:
            df_orig[column] = df_orig.apply(lambda row: cat_num(row, features.index(column), category_map, num_dist), axis=1)


        return df_orig

    
#function to return numerical embedding of categorical values
def cat_num_embed (df_orig, target, dist_metric):
    
    df = df_orig.copy()
    Y = df[target]
    Y = np.array(Y)
    df = df.drop(columns=[target], axis=1)
    
    # select numerical and categorical data 
    categorical_columns_selector = selector(dtype_include=object)
    categorical_columns = categorical_columns_selector(df)
    
    features = list(df.columns)
    
    if dist_metric == 'm_estimate':
        
        encoder = ce.MEstimateEncoder(cols=categorical_columns)
        
        x = df
        y = Y
        x_tf = encoder.fit_transform(x, y)
        
        col_embed = {}
        #diff = {}

        for val in categorical_columns:
            
######################## old ################################
            
#             col = np.unique(x[val], return_index=True)
#             col_dict = dict(zip(col[1], col[0]))

#             col_tf = np.unique(x_tf[val], return_index=True)
#             col_tf_dict = dict(zip(col_tf[1], col_tf[0]))

#             #dif = len(col[1]) - len(col_tf[1])

#             #diff[val] = dif

#             col_em = {}

#             for key in col_dict:
#                 if key in col_tf_dict:
#                     col_em[col_dict[key]] = col_tf_dict[key]


#             col_em = sorted(col_em.items(), key=lambda x:x[1])
#             col_em = dict(col_em)

###################### new #####################################

            col_em = pd.Series(x_tf[val].values,index=x[val]).to_dict()
    
            col_em = sorted(col_em.items(), key=lambda x:x[1])
            col_em = dict(col_em)

            col_embed[features.index(val)] = col_em
        
        return col_embed
    
    else:
    
        # get categorical features and apply labelencoding
        category_map = {}
        for f in categorical_columns:
            le = LabelEncoder()
            data_tmp = le.fit_transform(df[f].values)
            df[f] = data_tmp
            category_map[features.index(f)] = list(le.classes_)

        cat_vars = {}

        for f in categorical_columns:
            cat_vars[features.index(f)] = df[f].nunique()


        # convert df to numpy array
        df_np = df.to_numpy()

        disc_perc = [25, 50, 75]

        X_ord, cat_vars_ord = df_np, cat_vars

        # bin numerical features to compute the pairwise distance matrices
        cat_keys = list(cat_vars_ord.keys())
        n_ord = X_ord.shape[1]
        if len(cat_keys) != n_ord:
            fnames = [str(_) for _ in range(n_ord)]
            disc = Discretizer(X_ord, cat_keys, fnames, percentiles=disc_perc)
            X_bin = disc.discretize(X_ord)
            cat_vars_bin = {k: len(disc.names[k]) for k in range(n_ord) if k not in cat_keys}
        else:
            X_bin = X_ord
            cat_vars_bin = {}


        if dist_metric == 'abdm':
            num_dist = abdm_dist(X_bin, cat_vars_ord, cat_vars_bin)
        elif dist_metric == 'mvdm':
            num_dist = mvdm_dist(X_ord, Y, cat_vars_ord, alpha=1)


        cat_num = {}

        for column in categorical_columns:
            cat_col_idx = features.index(column)
            temp = {}
            for i in range(len(category_map[cat_col_idx])):
                temp[category_map[cat_col_idx][i]] = num_dist[cat_col_idx][i]

            sorted_dict = sorted(temp.items(), key=lambda x:x[1])
            sorted_dict = dict(sorted_dict)

            cat_num[cat_col_idx] = sorted_dict

        return cat_num
    
    
    
    
    
    

