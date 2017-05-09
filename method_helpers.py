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
        labels = kmeans_model.fit_predict(data)
        centroids = kmeans_model.cluster_centers_
        counter = Counter()
        for label in labels:
            counter[label] += 1
        print counter

    print "Estimating initial clusters..."
    i = 0
    clusters = {}
    for centroid in centroids:
        samples = [centroid]
        name = data_name + str(i)
        if config['distribution'] == 'gp':
            distribution = GaussianProcess(samples, name)
        elif config['distribution'] == 'gaussian':
            distribution = Gaussian(samples, name)
        clusters[name] = Cluster(distribution, name)
        i += 1

    return clusters, labels


@unpack_args
def m_step(cluster, iteration):
    if cluster.samples == []:
        return cluster
    cluster.reestimate(iteration)
    return cluster


@unpack_args
def e_step(sample, clusters):
    i = int(sample[0])
    sample = sample[1:]

    sample_likelihoods = []
    for cluster in clusters:  # find max likelihood cluster
        likelihood = cluster.likelihood(sample, range(len(sample)), i)
        sample_likelihoods.append(likelihood)

    max_likelihood = max(sample_likelihoods)
    max_index = sample_likelihoods.index(max_likelihood)

    return i, max_index


def assign_labeled_sample(sample, memberships, clusters):
    i = sample[0]
    sample = sample[1:]
    clusters[memberships[i]].assign_sample(sample, i)
