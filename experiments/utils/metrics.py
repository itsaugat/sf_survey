import numpy as np
import scipy as sp
from math import sqrt
from scipy.spatial import distance
from scipy.spatial.distance import cityblock


'''
Calculate L1 distance (Manhattan)
'''
def calculate_l1_distance(query, sf):
    return cityblock(query[0], sf)


'''
Calculate L2 distance (Euclidean)
'''
def calculate_l2_distance(query, sf):
    return distance.euclidean(query[0], sf)


'''
Calculate Mahalanobis distance between a point and the distribution
'''
def calculate_mahalanobis_dist(x=None, data=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution.
    """
    x_minus_mu = x - np.mean(data, axis=0)
    cov = np.cov(np.array(data).T)
    inv_covmat = sp.linalg.inv(cov)
    right_term = np.dot(x_minus_mu, inv_covmat)
    mahal_square = np.dot(right_term, x_minus_mu.T)
    return sqrt(mahal_square)
