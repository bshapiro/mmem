from collections import Counter
from cluster import Cluster
from distributions.gaussianprocess import GaussianProcess
from distributions.gaussian import Gaussian
from helpers import *
from sklearn.cluster import KMeans


def generate_initial_clusters(data, data_name):
    """
    Generates the initial clusters based on config.
    """
    if config['init'] is 'kmeans':
        kmeans_model = KMeans(config['k'], max_iter=1000)
        centroids = kmeans_model.cluster_centers_
    elif config['init'] is 'firstk':
        centroids = data[0:config['k'], :]

    print "Estimating initial clusters..."
    i = 0
    clusters = {}
    for centroid in centroids:
        samples = [np.reshape(centroid, (1, centroid.shape[0]))]
        name = data_name + str(i)
        if config['distribution'] == 'gp':
            distribution = GaussianProcess(samples, name)
        elif config['distribution'] == 'gaussian':
            distribution = Gaussian(samples, name)
        clusters[name] = Cluster(distribution, name)
        i += 1

    return clusters


@unpack_args
def parallel_m_step(cluster, iteration):
    if cluster.samples == []:
        return cluster
    cluster.reestimate(iteration)
    return cluster


@unpack_args
def parallel_e_step(sample, label, clusters):
    sample_likelihoods = []
    for cluster in clusters:  # find max likelihood cluster
        likelihood = cluster.likelihood(sample)
        sample_likelihoods.append(likelihood)

    max_likelihood = max(sample_likelihoods)
    max_index = sample_likelihoods.index(max_likelihood)

    return label, max_index


def assign_labeled_sample(sample, label, memberships, clusters):
    clusters[memberships[label]].assign_sample(sample, label)
