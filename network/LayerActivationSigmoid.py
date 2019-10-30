'''
Computes the logistic sigmoid activation function.
'''

from abc import abstractmethod
from Layer import Layer

import theano
import theano.tensor as T
import numpy as np
import math

class LayerActivationSigmoid(Layer):
    def __init__(self,
                 input_layer):
        super(LayerActivationSigmoid, self).__init__(input_layer, 'activation_sigmoid')

    def getTrainOutput(self):
        if self._input_layer.isOutputDistribution():
            print 'Warning: Variance approximation for sigmoid activation not tested'
            x_in_mean, x_in_var = self._input_layer.getTrainOutput()
            return LayerActivationSigmoid.sigmoidOfGaussian(x_in_mean, x_in_var)
        else:
            x_in = self._input_layer.getTrainOutput()
            x_out = T.nnet.sigmoid(x_in)
            return x_out
    
    def getPredictionOutput(self):
        x_in = self._input_layer.getPredictionOutput()
        return T.nnet.sigmoid(x_in)
    
    def getSampleOutput(self):
        x_in = self._input_layer.getSampleOutput()
        return T.nnet.sigmoid(x_in)
    
    def getOutputType(self):
        return 'real'
    
    def getMessage(self):
        return self._input_layer.getMessage() + '\n%20s: OutputShape=%15s, DistributionOutput=%d' % ('LayerActivationSgmd', str(self.getOutputShape()), self.isOutputDistribution())
    
    @staticmethod
    def sigmoidOfGaussian(m, v, eps=1e-6):
        # Gaussian approximation of the sigmoid applied to a Gaussian
        # We adapt approximation ideas from
        # Wang and Manning; Fast dropout training
        
        tanhSq_a = 4. - 2. * (2. ** 0.5)
        tanhSq_b = float(-np.log(2. ** 0.5 - 1))

        raise NotImplementedError('The mean and variance implementation for the sigmoid activation function has not been tested yet.')
        # Use identity sigmoid(x) = (tanh(x * 0.5) + 1) * 0.5 to transfer the implementation from LayerActivationTanh
        m = m * 0.5                # new compared to LayerActivationTanh
        v = v * 0.25               # new compared to LayerActivationTanh
        
        m_out = T.tanh(m * T.inv(T.sqrt(1. + float(np.pi) * 0.5 * v)))
        m_out = (m_out + 1.) * 0.5 # new compared to LayerActivationTanh
        v_out = 2. * (T.tanh((m - tanhSq_b * 0.5) * T.inv(T.sqrt(1. / tanhSq_a ** 2. + float(np.pi) * v * 0.5))) + 1.) \
            - T.sqr((T.tanh(m * T.inv(T.sqrt(1. + float(np.pi) * v * 0.5)))) + 1) + eps
        v_out = v_out * 0.25       # new compared to LayerActivationTanh

        return m_out, v_out
