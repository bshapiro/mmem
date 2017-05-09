from distribution import Distribution
import numpy as np
from scipy.stats import multivariate_normal


class Gaussian(Distribution):

    def __init__(self, samples, name):
        super(Gaussian, self).__init__(samples, name)

    def log_likelihood(self, sample):
        log_likelihood = self.gaussian.logpdf(sample)
        return log_likelihood

    def reestimate(self, samples):
        mean = np.mean(samples, 0)
        if len(samples) == 1:
            sigma = np.identity(mean.shape[0])
            self.gaussian = multivariate_normal(mean=mean, cov=sigma)
            return
        mean = np.mean(np.asarray(self.samples), 0)
        sigma = np.zeros((mean.shape[0], mean.shape[0]))
        for sample in samples:
            sigma += np.dot(sample - mean, np.transpose(sample - mean))
        sigma = sigma / len(self.samples)
        self.gaussian = multivariate_normal(mean=mean, cov=sigma)
