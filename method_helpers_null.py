from helpers import *


@unpack_args
def parallel_m_step(cluster, iteration):
    return cluster


@unpack_args
def parallel_e_step(sample, label, clusters):
    return label, 0
