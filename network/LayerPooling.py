'''
Pooling layer (max, sum, and average-pooling).
'''

from Layer import Layer

import numpy as np
import theano.tensor as T
import theano.tensor.signal.pool
import theano.tensor.nnet.neighbours # without this line it crashes when run on the cpu

class LayerPooling(Layer):
    def __init__(self,
                 input_layer,
                 ws,
                 stride=None,
                 mode='max',
                 ignore_border=False,
                 initial_parameters=None,
                 epsilon=1e-8):
        super(LayerPooling, self).__init__(input_layer, 'pooling')
        # The input must not be a feature map
        assert input_layer.isOutputFeatureMap() == True
        # mode must be in ['max', 'sum']
        assert mode in ['max', 'sum', 'avg']
        # ws must be a 2-tuple
        assert isinstance(ws, tuple) and len(ws) == 2

        self._mode = mode
        self._ws = ws
        self._stride = ws if stride is None else stride
        self._ignore_border = ignore_border
        self._epsilon = epsilon

    def getTrainOutput(self):
        if self._input_layer.isOutputDistribution():
            x_in_mean, x_in_var = self._input_layer.getTrainOutput()
            if self._mode == 'sum':
                x_out_mean = theano.tensor.signal.pool.pool_2d(x_in_mean, ws=self._ws, ignore_border=self._ignore_border, stride=self._stride, mode='sum')
                x_out_var = theano.tensor.signal.pool.pool_2d(x_in_var, ws=self._ws, ignore_border=self._ignore_border, stride=self._stride, mode='sum')
            elif self._mode == 'avg':
                inv_fan_in = T.inv(T.cast(self._ws[0] * self._ws[1], theano.config.floatX))
                x_out_mean = theano.tensor.signal.pool.pool_2d(x_in_mean, ws=self._ws, ignore_border=self._ignore_border, stride=self._stride, mode='sum') * inv_fan_in
                x_out_var = theano.tensor.signal.pool.pool_2d(x_in_var, ws=self._ws, ignore_border=self._ignore_border, stride=self._stride, mode='sum') * T.sqr(inv_fan_in)
            elif self._mode == 'max':
                if self._ws[0] != 2 or self._ws[1] != 2:
                    raise Exception('Pooling mode \'max\' with distributions is only supported for 2x2 downsampling')
                input_shape = self._input_layer.getOutputShape()
                mod_in_0 = input_shape[0] % self._ws[0]
                mod_in_1 = input_shape[1] % self._ws[1]
                if (mod_in_0 != 0 or mod_in_1 != 0) and self._ignore_border == False:
                    raise Exception('Pooling mode \'max\' with distributions supports only ignore_borders=True in the current implementation')

                _, rows, cols = self.getOutputShape()
                aux_shape = T.set_subtensor(x_in_mean.shape[2:], [rows, cols])
                print 'Debug: LayerPooling: ', self._input_layer.getOutputType()
                if self._input_layer.getOutputType() == 'binary':
                    # If the input to this layer is the sign function, we know
                    # that the distribution is a Bernoulli with values {-1,+1}.
                    # The probability that the maximum is -1 is equal to the
                    # product of the probabilities of each input being -1.
                    
#                     # Product implementation
#                     aux = T.nnet.neighbours.images2neibs((1. - x_in_mean) * 0.5, (2,2), mode='ignore_borders')
#                     aux = T.nnet.neighbours.neibs2images(T.prod(aux, axis=1, keepdims=True), (1,1), aux_shape)
#                     # aux now contains the probability of the maximum being -1
#                     x_out_mean = 1. - 2. * aux
#                     x_out_var = 1. - T.sqr(x_out_mean) + max(1e-6, self._epsilon)

#                     # New implementation
#                     # Product implementation
#                     aux_p = (1. - x_in_mean) * 0.5
#                     aux_p = T.maximum(aux_p, 1e-20)
#                     aux = T.nnet.neighbours.images2neibs(aux_p, (2,2), mode='ignore_borders')
#                     aux = T.nnet.neighbours.neibs2images(T.prod(aux, axis=1, keepdims=True), (1,1), aux_shape)
#                     x_out_mean = 1. - 2. * aux
#                     x_out_var = 1. - T.sqr(x_out_mean) + max(1e-6, self._epsilon)

                    # Sum-of-logarithms implementation: Can use pool_2d which is much faster
                    aux = T.log((1. - x_in_mean) * 0.5)
                    aux = theano.tensor.signal.pool.pool_2d(aux, ws=self._ws, ignore_border=self._ignore_border, stride=self._stride, mode='sum')
                    x_out_mean = 1. - 2. * T.exp(aux)
                    x_out_var = 1. - T.sqr(x_out_mean) + max(1e-6, self._epsilon)
                    
                    print 'Max-Pooling of binary values -> We can use this to compute maximum in closed form'
                elif self._input_layer.getOutputType() == 'binary01':
                    raise NotImplementedError()
                else:
                    assert self._ws == (2,2) and self._stride == (2,2)
                    aux_mean = T.nnet.neighbours.images2neibs(x_in_mean, (2,2), mode='ignore_borders')
                    aux_var = T.nnet.neighbours.images2neibs(x_in_var, (2,2), mode='ignore_borders')
                    g1_mean, g1_var = LayerPooling.maxOfGaussians(aux_mean[:,0], aux_var[:,0], aux_mean[:,1], aux_var[:,1], eps=self._epsilon)
                    g2_mean, g2_var = LayerPooling.maxOfGaussians(aux_mean[:,2], aux_var[:,2], aux_mean[:,3], aux_var[:,3], eps=self._epsilon)
                    g3_mean, g3_var = LayerPooling.maxOfGaussians(g1_mean, g1_var, g2_mean, g2_var, eps=self._epsilon)
                    x_out_mean = T.nnet.neighbours.neibs2images(g3_mean[:,None], (1,1), aux_shape)
                    x_out_var = T.nnet.neighbours.neibs2images(g3_var[:,None], (1,1), aux_shape)
            else:
                raise NotImplementedError('Unsupported pooling operation \'%s\'' % (self._mode))
            return x_out_mean, x_out_var
        else:
            x_in = self._input_layer.getTrainOutput()
            if self._mode in ['sum', 'max']:
                x_out = theano.tensor.signal.pool.pool_2d(x_in, ws=self._ws, ignore_border=self._ignore_border, stride=self._stride, mode=self._mode)
            elif self._mode == 'avg':
                inv_fan_in = T.inv(T.cast(self._ws[0] * self._ws[1], theano.config.floatX))
                x_out = theano.tensor.signal.pool.pool_2d(x_in, ws=self._ws, ignore_border=self._ignore_border, stride=self._stride, mode='sum') * inv_fan_in
            else:
                raise NotImplementedError('Unsupported pooling operation \'%s\'' % (self._mode))
            return x_out
    
    def getPredictionOutput(self):
        x_in = self._input_layer.getPredictionOutput()
        if self._mode in ['sum', 'max']:
            x_out = theano.tensor.signal.pool.pool_2d(x_in, ws=self._ws, ignore_border=self._ignore_border, stride=self._stride, mode=self._mode)
        elif self._mode == 'avg':
            inv_fan_in = T.inv(T.cast(self._ws[0] * self._ws[1], theano.config.floatX))
            x_out = theano.tensor.signal.pool.pool_2d(x_in, ws=self._ws, ignore_border=self._ignore_border, stride=self._stride, mode='sum') * inv_fan_in
        else:
            raise NotImplementedError('Unsupported pooling operation \'%s\'' % (self._mode))
        return x_out
    
    def getSampleOutput(self):
        x_in = self._input_layer.getSampleOutput()
        if self._mode in ['sum', 'max']:
            x_out = theano.tensor.signal.pool.pool_2d(x_in, ws=self._ws, ignore_border=self._ignore_border, stride=self._stride, mode=self._mode)
        elif self._mode == 'avg':
            inv_fan_in = T.inv(T.cast(self._ws[0] * self._ws[1], theano.config.floatX))
            x_out = theano.tensor.signal.pool.pool_2d(x_in, ws=self._ws, ignore_border=self._ignore_border, stride=self._stride, mode='sum') * inv_fan_in
        else:
            raise NotImplementedError('Unsupported pooling operation \'%s\'' % (self._mode))
        return x_out
    
    def getTrainUpdates(self):
        return self._input_layer.getTrainUpdates()
    
    def isOutputFeatureMap(self):
        return True
    
    def getOutputType(self):
        return self._input_layer.getOutputType()
    
    def getOutputShape(self):
        n_features, rows, cols = self._input_layer.getOutputShape()
        if self._ws == self._stride:
            if self._ignore_border:
                rows = rows // self._ws[0]
                cols = cols // self._ws[1]
            else:
                rows = (rows / self._ws[0]) if rows % self._ws[0] == 0 else (rows // self._ws[0] + 1)
                cols = (cols / self._ws[1]) if cols % self._ws[1] == 0 else (cols // self._ws[1] + 1)
        else:
            # TODO: Check if the following code also covers the case ws==stride,
            # so that we can basically shorten the code (check this)
            assert rows > self._ws[0] and cols > self._ws[1]
            if self._ignore_border:
                # Only compute how many pooling windows fit perfectly into the image
                rows = (rows - self._ws[0]) // self._stride[0] + 1
                cols = (cols - self._ws[1]) // self._stride[1] + 1
            else:
                # How many pooling windows fit perfectly
                rows_aux = (rows - self._ws[0]) // self._stride[0] + 1
                cols_aux = (cols - self._ws[1]) // self._stride[1] + 1
                # If some parts of the image do not fit perfectly, add 1 to the result
                if (rows - self._ws[0]) % self._stride[0] == 0:
                    rows = rows_aux
                else:
                    rows = rows_aux + 1
                if (cols - self._ws[1]) % self._stride[1] == 0:
                    cols = cols_aux
                else:
                    cols = cols_aux + 1

        return (n_features, rows, cols)
    
    def getMessage(self):
        return self._input_layer.getMessage() + '\n%20s: OutputShape=%15s, DistributionOutput=%d, ws=%s, mode=%s' % ('LayerPooling', str(self.getOutputShape()), self.isOutputDistribution(), str(self._ws), self._mode)

    @staticmethod
    def maxOfGaussians(m1, v1, m2, v2, eps=1e-8):
        # Gaussian approximation of the maximum of two Gaussians
        # Implementation according to:
        # Sinha et al.; Advances in Computation of the Maximum of a Set of Random Variables
        a_sqr = v1 + v2 + eps
        a = T.sqrt(a_sqr)
        alpha = (m1 - m2) / a

        aux_erf = T.erf(alpha * (0.5 ** 0.5))
        cdf_alpha_pos = 0.5 * (1. + aux_erf)
        cdf_alpha_neg = 0.5 * (1. - aux_erf)
        pdf_alpha = float(1. / (2. * np.pi)**0.5) * T.exp(-0.5 * T.sqr(alpha))
        a_times_pdf_alpha = a * pdf_alpha
         
        m_max = m1 * cdf_alpha_pos + m2 * cdf_alpha_neg + a_times_pdf_alpha
        v_max = (v1 + T.sqr(m1)) * cdf_alpha_pos \
              + (v2 + T.sqr(m2)) * cdf_alpha_neg \
              + (m1 + m2) * a_times_pdf_alpha \
              - T.sqr(m_max) + eps

        return m_max, v_max
