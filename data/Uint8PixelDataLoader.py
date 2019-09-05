import numpy as np

class Uint8PixelDataLoader(object):
    def __init__(self, x, t, batch_size, normalization='m1_p1', rng=None):
        assert normalization in ['none', 'm1_p1', 'zr_p1']
        assert x.dtype == np.uint8
        self._x = x
        self._t = t
        self._n_samples = self._x.shape[0]
        self._n_batches = self._n_samples // batch_size
        self._rnd = np.random.RandomState(1234) if rng is None else rng
        self._batch_idx = np.array_split(np.arange(self._n_samples), self._n_batches)
        self._batch_idx = [(idx[0], idx[-1] + 1) for idx in self._batch_idx]
        self._normalization = normalization
        
    def generateTrainData(self):
        randperm = self._rnd.permutation(self._n_samples)
        self._x = self._x[randperm]
        self._t = self._t[randperm]
        for idx in range(self._n_batches):
            if self._normalization == 'none':
                x_sample = np.asarray(self._x[self._batch_idx[idx][0]:self._batch_idx[idx][1]], dtype='float32')
            elif self._normalization == 'm1_p1':
                x_sample = (np.asarray(self._x[self._batch_idx[idx][0]:self._batch_idx[idx][1]], dtype='float32') * (2. / 255.)) - 1.
            elif self._normalization == 'zr_p1':
                x_sample = np.asarray(self._x[self._batch_idx[idx][0]:self._batch_idx[idx][1]], dtype='float32') * (1. / 255.)
            else:
                raise NotImplementedError()
            t_sample = self._t[self._batch_idx[idx][0]:self._batch_idx[idx][1]]
            yield x_sample, t_sample
        
    def generateTestData(self): # No random permutation here
        for idx in range(self._n_batches):
            if self._normalization == 'none':
                x_sample = np.asarray(self._x[self._batch_idx[idx][0]:self._batch_idx[idx][1]], dtype='float32')
            elif self._normalization == 'm1_p1':
                x_sample = np.asarray(self._x[self._batch_idx[idx][0]:self._batch_idx[idx][1]], dtype='float32') * (2. / 255.) - 1.
            elif self._normalization == 'zr_p1':
                x_sample = np.asarray(self._x[self._batch_idx[idx][0]:self._batch_idx[idx][1]], dtype='float32') * (1. / 255.)
            else:
                raise NotImplementedError()
            t_sample = self._t[self._batch_idx[idx][0]:self._batch_idx[idx][1]]
            yield x_sample, t_sample
        return
