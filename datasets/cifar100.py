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

def downloadCifar100(filename):
    url_cifar100 = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    tmp_cifar100 = 'cifar-100-python.tar.gz'

    print 'Downloading Cifar-100 dataset...'
    urlretrieve(url_cifar100, tmp_cifar100)
    
    print 'Uncompressing tar.gz files...'
    tar = taropen(tmp_cifar100, 'r:gz')
    tar.extract('cifar-100-python/train')
    tar.extract('cifar-100-python/test')
    tar.close()

    data_train = load_pickled('cifar-100-python/train')
    data_test = load_pickled('cifar-100-python/test')
    
    x_tr = np.stack(data_train['data'], axis=0)[:45000]
    t_tr = np.asarray(data_train['fine_labels'])[:45000]
    x_va = np.stack(data_train['data'], axis=0)[45000:]
    t_va = np.asarray(data_train['fine_labels'])[45000:]
    x_te = np.stack(data_test['data'], axis=0)
    t_te = np.asarray(data_test['fine_labels'])
    
    remove('cifar-100-python/train')
    remove('cifar-100-python/test')
    rmdir('cifar-100-python')
    remove(tmp_cifar100)
    
    print 'Storing Cifar-100 data to ''%s''' % (filename)
    np.savez_compressed(filename,
                        x_tr=x_tr, t_tr=t_tr,
                        x_va=x_va, t_va=t_va,
                        x_te=x_te, t_te=t_te)
    print 'Cifar-100 is now ready'
