import numpy as np

class Cifar10FlipShiftDataLoader(object):
    def __init__(self, x, t, batch_size, flip_axis=1, max_shift=4, requires_train=True, rng=None):
        self._x = x
        self._t = t
        self._n_samples = self._x.shape[0]
        self._n_batches = self._n_samples // batch_size
        self._rnd = np.random.RandomState(1234) if rng is None else rng
        self._batch_idx = np.array_split(np.arange(self._n_samples), self._n_batches)
        self._batch_idx = [(idx[0], idx[-1] + 1) for idx in self._batch_idx]
        if not isinstance(flip_axis, tuple):
            flip_axis = (flip_axis,)
        self._flip_axis = flip_axis
        if not isinstance(max_shift, tuple):
            max_shift = (max_shift, max_shift)
        self._max_shift = max_shift
        self._requires_train = True
        if self._requires_train:
            # Create padded training image only in case it is required
            self._x_tr = np.zeros((self._x.shape[0],
                                   self._x.shape[1], # should be 3
                                   self._x.shape[2] + 2 * self._max_shift[0],
                                   self._x.shape[3] + 2 * self._max_shift[1]),
                                  dtype=x.dtype)
            idx1_0 = self._max_shift[0]
            idx1_1 = idx1_0 + self._x.shape[2]
            idx2_0 = self._max_shift[1]
            idx2_1 = idx2_0 + self._x.shape[3]
            self._x_tr[:, :, idx1_0:idx1_1, idx2_0:idx2_1] = self._x.copy()
            self._t_tr = self._t.copy() 
        
    def generateTrainData(self):
        assert self._requires_train
        randperm = self._rnd.permutation(self._n_samples)
        self._x_tr = self._x_tr[randperm]
        self._t_tr = self._t_tr[randperm]
        for idx in range(self._n_batches):
            x_sample = self._x_tr[self._batch_idx[idx][0]:self._batch_idx[idx][1]]
            t_sample = self._t_tr[self._batch_idx[idx][0]:self._batch_idx[idx][1]]

            x_sample_transformed = np.zeros((x_sample.shape[0],
                                             self._x.shape[1],
                                             self._x.shape[2],
                                             self._x.shape[3]), dtype=x_sample.dtype)
            # Generate random crop
            for idx_sample in range(x_sample.shape[0]):
                img = x_sample[idx_sample]

                # Random crop
                s1 = self._rnd.randint(0, 2 * self._max_shift[0] + 1)
                s2 = self._rnd.randint(0, 2 * self._max_shift[1] + 1)
                img = img[:, s1:s1+self._x.shape[2], s2:s2+self._x.shape[3]]

                # img[0]: color
                # img[1]: row
                # img[2]: column
                flip_bits = self._rnd.randint(0, 2, (2,))
                if 0 in self._flip_axis and flip_bits[0] == 1:
                    # vertical flip
                    img = img[:, ::-1, :]
                if 1 in self._flip_axis and flip_bits[1] == 1:
                    # horizontal flip
                    img = img[:, :, ::-1]
                
                x_sample_transformed[idx_sample, ...] = img

            yield x_sample_transformed, t_sample
        
    def generateTestData(self): # No random permutation here
        for idx in range(self._n_batches):
            x_sample = self._x[self._batch_idx[idx][0]:self._batch_idx[idx][1]]
            t_sample = self._t[self._batch_idx[idx][0]:self._batch_idx[idx][1]]
            yield x_sample, t_sample
        return