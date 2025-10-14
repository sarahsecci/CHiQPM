from statistics import NormalDist

import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import trange


def get_gmm(feature, n_components):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(feature.reshape(-1, 1))
    return gmm

def gmm_overlap_per_feature(features):
    n_features = features.shape[1]
    answer = np.zeros(n_features)
    for i in trange(n_features):
            answer[i] = gmm_overlap(features[:, i])
    return answer

def gmm_overlap(feature):
    gmm =get_gmm(feature, n_components=2)
    overlap = get_overlap(gmm)
    return overlap

def get_overlap(mixture_model):
    return NormalDist(mu=mixture_model.means_[0], sigma=np.sqrt(mixture_model.covariances_[0])).overlap(
        NormalDist(mu=mixture_model.means_[1], sigma=np.sqrt(mixture_model.covariances_[1])))




def gmm_metrics(features):
    n_features = features.shape[1]
    overlaps = np.zeros(n_features)
    means = np.zeros(n_features)
    variances = np.zeros(n_features)
    for i in trange(n_features):
        this_feature = features[:, i]
        gmm = get_gmm(this_feature, n_components=2)
        overlap = get_overlap(gmm)
        active_arg = np.argmax(gmm.means_)
        means[i] = gmm.means_[active_arg]
        variances[i] = gmm.covariances_[active_arg]
        overlaps[i] = overlap
    return overlaps, means, variances, np.var(means)