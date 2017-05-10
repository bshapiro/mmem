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
from optparse import OptionParser
from sklearn.preprocessing import scale
import math
import numpy as np
import pandas as pd
import time


def e_step(samples, clusters, memberships, iteration):
    print "Running iteration ", iteration
    print "Running E step..."

    reassigned_samples = 0
    labels = range(len(samples))
    if config['parallel']:
        pool = Pool(processes=config['n_processes'], maxtasksperchild=100)
        new_memberships = dict(pool.map(parallel_e_step, zip(samples, labels, repeat(clusters))))
        pool.close()
        pool.join()
        for key, value in new_memberships.items():
            if new_memberships[key] != memberships.get(key):
                reassigned_samples += 1
        memberships = new_memberships
    else:
        for i in range(len(samples)):  # iterate through samples

            sample = samples[i]

            sample_likelihoods = []
            for cluster in clusters:  # find max likelihood cluster
                likelihood = cluster.likelihood(sample)
                sample_likelihoods.append(likelihood)

            max_likelihood = max(sample_likelihoods)
            max_index = sample_likelihoods.index(max_likelihood)

            if memberships.get(i) != max_index:
                reassigned_samples += 1  # keep track of how many are reassigned
                memberships[i] = max_index

    #################### Hardcoded assignments for testing ####################
    num_samples_per_cluster = int(math.floor(len(samples) / len(clusters)))
    for k in range(len(clusters)):
        for i in range(num_samples_per_cluster):
            memberships[k*num_samples_per_cluster + i] = k
    ###########################################################################

    return memberships, reassigned_samples


def m_step(clusters, iteration):
    print "Running M step..."
    if config['parallel']:
        pool = Pool(processes=config['n_processes'], maxtasksperchild=1)
        clusters = pool.map(parallel_m_step, zip(clusters, [iteration]*len(clusters)))
        pool.close()
        pool.join()
    else:
        for cluster in clusters:  # reestimate all of the clusters
            cluster.reestimate(iteration)


def run_em(samples, clusters):
    if config['parallel']:
        print "Using max of " + str(cpu_count()) + " processes."

    memberships = {}
    iterations = 0

    for iteration in range(10):  # run 10 iterations

        for cluster in clusters:  # unassign any samples assigned to clusters
            cluster.clear_samples()

        memberships, reassigned_samples = e_step(samples, clusters, memberships, iteration)
        map(lambda (sample, label): assign_labeled_sample(sample, label, memberships, clusters), zip(samples, range(len(samples))))

        # test convergence
        print "Reassigned samples: ", reassigned_samples
        if reassigned_samples < 0.05*len(samples):
            break

        m_step(clusters, iteration)

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

    print '************'
    begin = time.time()

    if options.num_processes is not None:
        config['n_processes'] = int(options.num_processes)
        config['parallel'] = options.parallel
    data = pd.read_csv(options.filename, sep=',').as_matrix()
    print "Shape:", data.shape

    data = scale(data.T, with_mean=True, with_std=True).T
    # data = data[:50, :]
    samples = [np.reshape(data[i], (1, len(data[i]))) for i in range(data.shape[0])]

    clusters = generate_initial_clusters(data, config['data_name'])
    clusters, memberships = run_em(samples, clusters.values())

    dump(memberships, open(generate_output_dir() + 'memberships.dump', 'w'))  # dump memberships for further analysis

    end = time.time()

    print '************'
    print "Elapsed time: ", end-begin
