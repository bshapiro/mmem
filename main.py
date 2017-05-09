###### MUST COME FIRST ######
import matplotlib as mpl
mpl.use('Agg')
#############################

from config import config
from cPickle import dump
from helpers import *
from itertools import repeat
from method_helpers import *
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
from optparse import OptionParser


def e_step(data, clusters, labels, memberships, iteration):
    print "Running iteration ", iteration
    print "Running E step..."

    reassigned_samples = 0
    if config['parallel']:
        data_labeled = np.column_stack((range(data.shape[0]), data))
        pool = Pool(processes=config['n_processes'], maxtasksperchild=100)
        new_memberships = dict(pool.map(e_step, zip(data_labeled, repeat(clusters))))
        pool.close()
        pool.join()
        for key, value in new_memberships.items():
            if new_memberships[key] != memberships.get(key):
                reassigned_samples += 1
        memberships = new_memberships
        map(lambda sample: assign_labeled_sample(sample, memberships, clusters), data_labeled)
    else:
        for i in range(data.shape[0]):  # iterate through samples

            sample = data[i]

            sample_likelihoods = []
            for cluster in clusters:  # find max likelihood cluster
                likelihood = cluster.likelihood(sample)
                sample_likelihoods.append(likelihood)

            max_likelihood = max(sample_likelihoods)
            max_index = sample_likelihoods.index(max_likelihood)

            clusters[max_index].assign_sample(sample, i)  # assign samples to clusters
            if memberships.get(i) != max_index:
                reassigned_samples += 1  # keep track of how many are reassigned
                memberships[i] = max_index

    return memberships, reassigned_samples


def m_step(data, clusters, labels):
    print "Running M step..."
    if config['parallel']:
        pool = Pool(processes=config['n_processes'], maxtasksperchild=1)
        clusters = pool.map(m_step, zip(clusters, [iteration]*len(clusters)))
        pool.close()
        pool.join()
    else:
        for cluster in clusters:  # reestimate all of the clusters
            cluster.reestimate(iteration)


def run_em(data, clusters, labels):
    if config['parallel']:
        print "Using max of " + str(cpu_count()) + " processes."

    memberships = {}
    iterations = 0

    for iteration in range(100):  # max 100 iterations, but we never hit this number

        for cluster in clusters:  # unassign any samples assigned to clusters
            cluster.clear_samples()

        memberships, reassigned_samples = e_step(data, clusters, labels, memberships, iteration)

        # test convergence
        print "Reassigned samples: ", reassigned_samples
        if reassigned_samples < 0.05*data.shape[0]:
            break

        m_step(data, clusters, labels)

        iterations += 1

    print "Converged in ", iterations, " iterations."
    return clusters, memberships


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-d", "--data", dest="filename",
                      help="Location of data", metavar="FILE")
    parser.add_option("-n", "--processes", dest="num_processes",
                      help="Number of processes")
    parser.add_option("-p", "--parallel", dest="parallel",
                      action="store_true", default=False)
    (options, args) = parser.parse_args()

    config['n_processes'] = options.num_processes
    config['parallel'] = options.parallel
    data = pd.read_csv(options.filename, sep=',').as_matrix()

    print "Shape:", data.shape

    data = scale(data.T, with_mean=True, with_std=True).T

    clusters, labels, init_likelihood = generate_initial_clusters(data, config['data_name'])
    clusters, memberships, likelihoods = run_em(data, clusters.values(), labels)

    dump(memberships, open(generate_output_dir() + 'memberships.dump', 'w'))  # dump memberships for further analysis
