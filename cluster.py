import numpy as np


class Cluster:

    def __init__(self, distribution, name, link=None):
        self.distribution = distribution
        self.name = name
        self.samples = []
        self.sample_indices = []
        self.link = None

    def likelihood(self, sample, index):
        log_likelihood = self.distribution.log_likelihood(sample)
        return log_likelihood

    def assign_sample(self, sample, index):
        self.samples.append(sample)
        self.sample_indices.append(index)

    def contains_sample(self, index):
        return index in set(self.sample_indices)

    def clear_samples(self):
        self.samples = []
        self.sample_indices = []

    def reestimate(self, iteration):
        print 'Cluster', self.name + ';\t', '# of samples:', len(self.samples)
        self.distribution.reestimate(self.samples)
