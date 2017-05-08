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


def run_em(data, clusters, labels):
    if config['parallel']:
        print "Using max of " + str(cpu_count()) + " processes."

    memberships = {}
    iterations = 0
    likelihoods = []

    for iteration in range(100):  # max 100 iterations, but we never hit this number

        for cluster in clusters:  # unassign any samples assigned to clusters
            cluster.clear_samples()

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
                sample = np.reshape(sample, (len(sample), 1))

                sample_likelihoods = []
                for cluster in clusters:  # find max likelihood cluster
                    likelihood = cluster.likelihood(sample, range(len(sample)), i)
                    sample_likelihoods.append(likelihood)

                max_likelihood = max(sample_likelihoods)
                max_index = sample_likelihoods.index(max_likelihood)

                clusters[max_index].assign_sample(sample, i)  # assign samples to clusters
                if memberships.get(i) != max_index:
                    reassigned_samples += 1  # keep track of how many are reassigned
                    memberships[i] = max_index

        # test convergence
        print "Reassigned samples: ", reassigned_samples
        if reassigned_samples < 0.05*data.shape[0]:
            break

        e_likelihood = likelihood_for_clusters(clusters)
        print "Likelihood after E step:", e_likelihood
        likelihoods.append(e_likelihood)
        print "Running M step..."
        if config['parallel']:
            pool = Pool(processes=config['n_processes'], maxtasksperchild=1)
            clusters = pool.map(m_step, zip(clusters, [iteration]*len(clusters)))
            pool.close()
            pool.join()
        else:
            for cluster in clusters:  # reestimate all of the clusters
                if cluster.samples == []:
                    continue
                cluster.reestimate(iteration)

        m_likelihood = likelihood_for_clusters(clusters)
        print "Likelihood after M step:", m_likelihood
        likelihoods.append(m_likelihood)

        iterations += 1

    print "Converged in ", iterations, " iterations."
    print "Number of reassigned samples in last iteration: ", reassigned_samples

    return clusters, memberships, likelihoods

if __name__ == "__main__":
    config['views'] = 'single'
    print "Shape:", data.shape
    data = scale(data.T, with_mean=True, with_std=True).T

    clusters, labels, init_likelihood = generate_initial_clusters(data, config['dataset'])
    clusters, memberships, likelihoods = run_em(data, clusters.values(), labels)
    # likelihoods.insert(0, init_likelihood)

    dump(memberships, open(generate_output_dir() + 'memberships.dump', 'w'))  # dump memberships for further analysis
