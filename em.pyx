from libc.stdlib cimport malloc, free
import ctypes
from cython.parallel import parallel, prange


def e_step(samples, clusters, memberships, iteration):
    print "Running iteration ", iteration
    print "Running E step..."

    cdef int num_samples = len(samples)
    cdef int num_clusters = len(clusters)
    cdef int j
    cdef int max_index
    cdef float max_likelihood
    cdef float likelihood
    cdef int reassigned_samples = 0

    cdef int* new_memberships = <int *>malloc(num_samples * sizeof(int))
    cdef float* sample_likelihoods = <float *>malloc(num_clusters * sizeof(float))

    with nogil, parallel():
        for i in prange(num_samples):  # iterate through samples
            j = 0
            max_index = 0
            max_likelihood = -999999999999
            for cluster in clusters:  # find max likelihood cluster
                likelihood = cluster.likelihood(samples[i])
                sample_likelihoods[j] = likelihood
                if likelihood > max_likelihood:
                    max_likelihood = likelihood
                    max_index = j
                j = j + 1
            j = 0

            if memberships[i] != max_index:
                reassigned_samples = reassigned_samples + 1  # keep track of how many are reassigned
                new_memberships[i] = max_index

    # TODO: convert new_memberships to numpy array so that we're returning a python object
    memberships = new_memberships
    return new_memberships, reassigned_samples
