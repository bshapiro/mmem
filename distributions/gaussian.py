from distribution import Distribution
from GP import fit_gp
import numpy as np


class Gaussian(Distribution):
    """
    Expects each sample to be in form [x, y], and set of samples to be in form [x], [y].
    """

    def __init__(self, samples):
        super(self.__class__, self).__init__(samples)
        self.mean = None
        self.sigma = None

    def log_likelihood(self, sample):
        log_likelihood = np.random.multivariate_normal(sample, self.mean, self.sigma)
        return log_likelihood

    def reestimate(self, samples):
        self.mean = np.mean(self.samples)
        self.sigma = np.zeros((self.mean.shape[0], self.mean.shape[0]))
        for sample in samples:
            self.sigma += np.dot(sample - self.mean, np.transpose(sample - self.mean))
        self.sigma = self.sigma / len(self.samples)
        
        
