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
        mean = np.mean(np.asarray(samples), 0)[0]
        if len(samples) == 1:
            sigma = np.identity(mean.shape[0]) * 1.0001
            self.gaussian = multivariate_normal(mean=mean, cov=sigma)
            return
        sigma = np.zeros((mean.shape[0], mean.shape[0]))
        for sample in samples:
            sigma += np.dot(np.transpose(sample - mean), sample - mean)
        sigma = sigma / len(samples) + np.eye(mean.shape[0])*0.00001
        try:
            self.gaussian = multivariate_normal(mean=mean, cov=sigma)
        except:
            import pdb; pdb.set_trace()
            
