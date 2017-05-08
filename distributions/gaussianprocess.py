from distribution import Distribution
from GP import fit_gp
import numpy as np


class GaussianProcess(Distribution):
    """
    Expects each sample to be in form [x, y], and set of samples to be in form [x], [y].
    """

    def __init__(self, samples):
        super(self.__class__, self).__init__(samples)

    def log_likelihood(self, sample):
        x = sample[0]
        y = sample[1]
        x = np.reshape(np.asarray(x), (len(x), 1))
        y = np.reshape(np.asarray(y), (len(y), 1))
        self.gp.set_XY(x, y)
        log_likelihood = self.gp.log_likelihood()
        return log_likelihood

    def reestimate(self, samples):
        x = samples[0]
        y = samples[1]
        y = np.asarray(y).reshape(len(y) * y[0].shape[0])
        x = np.asarray(x).reshape(len(x) * x[0].shape[0])
        self.gp = fit_gp(y, x, self.name)
