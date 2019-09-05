import numpy as np
from urllib import urlretrieve
from tarfile import open as taropen
from cPickle import load as pickleload
from os import remove, rmdir

# The following function is essentially taken from https://www.cs.toronto.edu/~kriz/cifar.html
def load_pickled(filename):
    with open(filename, 'rb') as f:
        dct = pickleload(f)
    return dct

def downloadCifar10(filename):
    url_cifar10 = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    tmp_cifar10 = 'cifar-10-python.tar.gz'

    print 'Downloading Cifar-10 dataset...'
    urlretrieve(url_cifar10, tmp_cifar10)
    
    print 'Uncompressing tar.gz files...'
    tar = taropen(tmp_cifar10, 'r:gz')
    tar.extract('cifar-10-batches-py/data_batch_1')
    tar.extract('cifar-10-batches-py/data_batch_2')
    tar.extract('cifar-10-batches-py/data_batch_3')
    tar.extract('cifar-10-batches-py/data_batch_4')
    tar.extract('cifar-10-batches-py/data_batch_5')
    tar.extract('cifar-10-batches-py/test_batch')
    tar.close()

    data_batch_1 = load_pickled('cifar-10-batches-py/data_batch_1')
    data_batch_2 = load_pickled('cifar-10-batches-py/data_batch_2')
    data_batch_3 = load_pickled('cifar-10-batches-py/data_batch_3')
    data_batch_4 = load_pickled('cifar-10-batches-py/data_batch_4')
    data_batch_5 = load_pickled('cifar-10-batches-py/data_batch_5')
    data_test = load_pickled('cifar-10-batches-py/test_batch')
    
    x_tr = np.concatenate((np.stack(data_batch_1['data'], axis=0),
                           np.stack(data_batch_2['data'], axis=0),
                           np.stack(data_batch_3['data'], axis=0),
                           np.stack(data_batch_4['data'], axis=0),
                           np.stack(data_batch_5['data'], axis=0)[:5000]), axis=0)
    t_tr = np.concatenate((np.asarray(data_batch_1['labels']),
                           np.asarray(data_batch_2['labels']),
                           np.asarray(data_batch_3['labels']),
                           np.asarray(data_batch_4['labels']),
                           np.asarray(data_batch_5['labels'])[:5000]), axis=0)
    x_va = np.stack(data_batch_5['data'], axis=0)[5000:]
    t_va = np.asarray(data_batch_5['labels'])[5000:]
    x_te = np.stack(data_test['data'], axis=0)
    t_te = np.asarray(data_test['labels'])
    
    remove('cifar-10-batches-py/data_batch_1')
    remove('cifar-10-batches-py/data_batch_2')
    remove('cifar-10-batches-py/data_batch_3')
    remove('cifar-10-batches-py/data_batch_4')
    remove('cifar-10-batches-py/data_batch_5')
    remove('cifar-10-batches-py/test_batch')
    rmdir('cifar-10-batches-py')
    remove(tmp_cifar10)
    
    print 'Storing Cifar-10 data to ''%s''' % (filename)
    np.savez_compressed(filename,
                        x_tr=x_tr, t_tr=t_tr,
                        x_va=x_va, t_va=t_va,
                        x_te=x_te, t_te=t_te)
    print 'Cifar-10 is now ready'
