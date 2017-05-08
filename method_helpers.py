from collections import Counter
from cluster import Cluster
from distributions.distribution import Distribution
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
        distribution = Distribution(samples, name)
        clusters[name] = Cluster(distribution, name)
        i += 1

    init_likelihood = likelihood_given_init_clusters(data, labels, clusters, data_name)
    print "Initial likelihood:", init_likelihood

    return clusters, labels, init_likelihood


def likelihood_given_init_clusters(data, labels, clusters, data_name):
    """
    Calculate the likelihood of the data given the initial cluster choices.
    """
    total_likelihood = 0
    index = 0
    for sample in data:
        label = labels[index]
        if label == -1:
            max_likelihood = max([gp_cluster.likelihood(sample, range(len(sample)), index) for gp_cluster in clusters.values() if gp_cluster.name.startswith(data_name)])
            total_likelihood += max_likelihood
        else:
            cluster_name = data_name + str(label)
            total_likelihood += clusters[cluster_name].likelihood(sample, range(len(sample)), index)
        index += 1
    return total_likelihood


def likelihood_for_clusters(clusters):
    """
    Calculate the likelihood of all the data assigned to their clusters.
    """
    total_likelihood = 0
    for cluster in clusters:
        if len(cluster.samples) != 0:
            x = range(len(cluster.samples[0]))
            index = 0
            for sample in cluster.samples:
                total_likelihood += cluster.likelihood(sample, x, index)
                index += 1
    return total_likelihood


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
