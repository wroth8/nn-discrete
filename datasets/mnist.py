'''
@author Wolfgang Roth
'''

import numpy as np

from os import remove
from urllib import urlretrieve
from shutil import copyfileobj
from gzip import open as gzopen
from struct import unpack

def downloadMnist(filename):
    '''
    Downloads the MNIST data set and stores it to the given file in npz format
    '''
    url_xTrain = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    url_tTrain = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    url_xTest = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    url_tTest = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    
    tmp_xTrain = 'tmp_mnist_xTrain.gz'
    tmp_tTrain = 'tmp_mnist_tTrain.gz'
    tmp_xTest = 'tmp_mnist_xTest.gz'
    tmp_tTest = 'tmp_mnist_tTest.gz'

    print 'Downloading MNIST train images...'
    urlretrieve(url_xTrain, tmp_xTrain)
    print 'Download MNIST train labels...'
    urlretrieve(url_tTrain, tmp_tTrain)
    print 'Download MNIST test images...'
    urlretrieve(url_xTest, tmp_xTest)
    print 'Download MNIST test labels...'
    urlretrieve(url_tTest, tmp_tTest)
    print 'Downloading finished'
    
    print 'Uncompressing gz files...'
    with gzopen(tmp_xTrain, 'rb') as f_in, open(tmp_xTrain[:-3], 'wb') as f_out:
        copyfileobj(f_in, f_out)
    with gzopen(tmp_tTrain, 'rb') as f_in, open(tmp_tTrain[:-3], 'wb') as f_out:
        copyfileobj(f_in, f_out)
    with gzopen(tmp_xTest, 'rb') as f_in, open(tmp_xTest[:-3], 'wb') as f_out:
        copyfileobj(f_in, f_out)
    with gzopen(tmp_tTest, 'rb') as f_in, open(tmp_tTest[:-3], 'wb') as f_out:
        copyfileobj(f_in, f_out)
    
    print 'Loading uncompressed data...'
    with open(tmp_xTrain[:-3], 'rb') as f:
        magic, nImages, nRows, nCols = unpack('>IIII', f.read(16))
        assert magic == 2051
        assert nImages == 60000
        assert nRows == 28
        assert nCols == 28
        x_tr_np = np.fromfile(f, dtype=np.uint8).reshape(nImages, nRows*nCols).astype(np.float32) / 255.
   
    with open(tmp_tTrain[:-3], 'rb') as f:
        magic, nImages = unpack('>II', f.read(8))
        assert magic == 2049
        assert nImages == 60000
        t_tr_np = np.fromfile(f, dtype=np.int8)

    with open(tmp_xTest[:-3], 'rb') as f:
        magic, nImages, nRows, nCols = unpack('>IIII', f.read(16))
        assert magic == 2051
        assert nImages == 10000
        assert nRows == 28
        assert nCols == 28
        x_te_np = np.fromfile(f, dtype=np.uint8).reshape(nImages, nRows*nCols).astype(np.float32) / 255.
        
    with open(tmp_tTest[:-3], 'rb') as f:
        magic, nImages = unpack('>II', f.read(8))
        assert magic == 2049
        assert nImages == 10000
        t_te_np = np.fromfile(f, dtype=np.int8)
        
    x_va_np = x_tr_np[50000:]
    t_va_np = t_tr_np[50000:]
    x_tr_np = x_tr_np[:50000]
    t_tr_np = t_tr_np[:50000]

    print 'Removing temporary files...'
    remove(tmp_xTrain)
    remove(tmp_xTrain[:-3])
    remove(tmp_tTrain)
    remove(tmp_tTrain[:-3])
    remove(tmp_xTest)
    remove(tmp_xTest[:-3])
    remove(tmp_tTest)
    remove(tmp_tTest[:-3])
    
    print 'Storing MNIST data to ''%s''' % (filename)
    np.savez_compressed(filename,
                        x_tr=x_tr_np, t_tr=t_tr_np,
                        x_va=x_va_np, t_va=t_va_np,
                        x_te=x_te_np, t_te=t_te_np)

    print 'MNIST is now ready'
