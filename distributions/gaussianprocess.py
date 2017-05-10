from distribution import Distribution
from GP import fit_gp
import numpy as np


class GaussianProcess(Distribution):

    def __init__(self, samples, name):
        super(GaussianProcess, self).__init__(samples, name)

    def log_likelihood(self, sample):
        x = range(len(sample))
        y = sample
        x = np.reshape(np.asarray(x), (len(x), 1))
        self.gp.set_XY(x, y)
        log_likelihood = self.gp.log_likelihood()
        return log_likelihood

    def reestimate(self, samples):
        x = range(samples[0].shape[1])
        y = samples
        y = np.asarray(y).reshape(len(y) * y[0].shape[1])
        x = np.asarray(x*len(samples))
        self.gp = fit_gp(y, x, self.name)
