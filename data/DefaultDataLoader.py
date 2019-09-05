import numpy as np

class DefaultDataLoader(object):
    def __init__(self, x, t, batch_size, rng=None):
        self._x = x
        self._t = t
        self._n_samples = self._x.shape[0]
        self._n_batches = self._n_samples // batch_size
        self._rnd = np.random.RandomState(1234) if rng is None else rng
        self._batch_idx = np.array_split(np.arange(self._n_samples), self._n_batches)
        self._batch_idx = [(idx[0], idx[-1] + 1) for idx in self._batch_idx]
        
    def generateTrainData(self):
        randperm = self._rnd.permutation(self._n_samples)
        self._x = self._x[randperm]
        self._t = self._t[randperm]
        for idx in range(self._n_batches):
            x_sample = self._x[self._batch_idx[idx][0]:self._batch_idx[idx][1]]
            t_sample = self._t[self._batch_idx[idx][0]:self._batch_idx[idx][1]]
            yield x_sample, t_sample
        
    def generateTestData(self): # No random permutation here
        for idx in range(self._n_batches):
            x_sample = self._x[self._batch_idx[idx][0]:self._batch_idx[idx][1]]
            t_sample = self._t[self._batch_idx[idx][0]:self._batch_idx[idx][1]]
            yield x_sample, t_sample
        return