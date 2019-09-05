import numpy as np
from urllib import urlretrieve
from scipy.io import loadmat
from os import remove


def downloadSvhn(filename, remove_tmp_files=True):
    url_svhn_train = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
    url_svhn_train_extra = 'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat'
    url_svhn_test = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
    tmp_train = 'train_32x32.mat'
    tmp_train_extra = 'extra_32x32.mat'
    tmp_test = 'test_32x32.mat'

    print 'Downloading SVHN train images...'
    urlretrieve(url_svhn_train, tmp_train)
    print 'Downloading SVHN extra train images...'
    urlretrieve(url_svhn_train_extra, tmp_train_extra)
    print 'Downloading SVHN test images...'
    urlretrieve(url_svhn_test, tmp_test)
    
    data_train = loadmat(tmp_train)
    data_train_extra = loadmat(tmp_train_extra)
    data_test = loadmat(tmp_test)
    
    data_train['X'] = data_train['X'].transpose(3, 2, 0, 1)
    data_train['y'] = data_train['y'].flatten() - 1
    data_train_extra['X'] = data_train_extra['X'].transpose(3, 2, 0, 1)
    data_train_extra['y'] = data_train_extra['y'].flatten() - 1
    data_test['X'] = data_test['X'].transpose(3, 2, 0, 1)
    data_test['y'] = data_test['y'].flatten() - 1

    # Use the method described in [1] to divide the training set into a training and validation set, respectively. That
    # is, we use the last 400 samples per class from train and the last 200 samples per class from train_extra and put
    # them into the validation set.
    # [1] Sermanet, P., Chintala, S., LeCun, Y.: Convolutional neural networks applied to house numbers digit
    # classification. ICPR, 2012
    x_tr = np.zeros((0,3,32,32), dtype=np.uint8)
    t_tr = np.zeros((0,), dtype=np.uint8)
    x_va = np.zeros((0,3,32,32), dtype=np.uint8)
    t_va = np.zeros((0,), dtype=np.uint8)
    for c in range(10):
        c_idx, = (data_train['y'] == c).nonzero()
        c_idx_tr = c_idx[:-400]
        c_idx_va = c_idx[-400:]
        x_tr = np.concatenate([x_tr, data_train['X'][c_idx_tr]], axis=0)
        t_tr = np.concatenate([t_tr, data_train['y'][c_idx_tr]], axis=0)
        x_va = np.concatenate([x_va, data_train['X'][c_idx_va]], axis=0)
        t_va = np.concatenate([t_va, data_train['y'][c_idx_va]], axis=0)
        
        c_idx, = (data_train_extra['y'] == c).nonzero()
        c_idx_tr = c_idx[:-200]
        c_idx_va = c_idx[-200:]
        x_tr = np.concatenate([x_tr, data_train_extra['X'][c_idx_tr]], axis=0)
        t_tr = np.concatenate([t_tr, data_train_extra['y'][c_idx_tr]], axis=0)
        x_va = np.concatenate([x_va, data_train_extra['X'][c_idx_va]], axis=0)
        t_va = np.concatenate([t_va, data_train_extra['y'][c_idx_va]], axis=0)

    # Randomly permute x_tr and x_va since they are now ordered according to class labels
    rng = np.random.RandomState(1234)
    randperm_tr = rng.permutation(x_tr.shape[0])
    x_tr = x_tr[randperm_tr]
    t_tr = t_tr[randperm_tr]
    randperm_va = rng.permutation(x_va.shape[0])
    x_va = x_va[randperm_va]
    t_va = t_va[randperm_va]
        
    x_te = data_test['X']
    t_te = data_test['y']

    if remove_tmp_files:
        remove(tmp_train)
        remove(tmp_train_extra)
        remove(tmp_test)

    print 'Storing SVHN data to ''%s''' % (filename)
    np.savez_compressed(filename,
                        x_tr=x_tr, t_tr=t_tr,
                        x_va=x_va, t_va=t_va,
                        x_te=x_te, t_te=t_te)
    print 'SVHN is now ready'
