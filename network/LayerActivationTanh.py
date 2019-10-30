'''
Computes the tanh activation function.
'''

from abc import abstractmethod
from Layer import Layer

import theano
import theano.tensor as T
import numpy as np
import math

class LayerActivationTanh(Layer):
    def __init__(self,
                 input_layer):
        super(LayerActivationTanh, self).__init__(input_layer, 'activation_tanh')

    def getTrainOutput(self):
        if self._input_layer.isOutputDistribution():
            print 'Warning: Variance approximation for tanh activation not tested'
            x_in_mean, x_in_var = self._input_layer.getTrainOutput()
            return LayerActivationTanh.tanhOfGaussian(x_in_mean, x_in_var)
        else:
            x_in = self._input_layer.getTrainOutput()
            x_out = T.tanh(x_in)
            return x_out
    
    def getPredictionOutput(self):
        x_in = self._input_layer.getPredictionOutput()
        return T.tanh(x_in)
    
    def getSampleOutput(self):
        x_in = self._input_layer.getSampleOutput()
        return T.tanh(x_in)
    
    def getOutputType(self):
        return 'real'
    
    def getMessage(self):
        return self._input_layer.getMessage() + '\n%20s: OutputShape=%15s, DistributionOutput=%d' % ('LayerActivationTanh', str(self.getOutputShape()), self.isOutputDistribution())
    
    @staticmethod
    def tanhOfGaussian(m, v, eps=1e-6):
        # Gaussian approximation of the tanh applied to a Gaussian
        # We adapt approximation ideas from
        # Wang and Manning; Fast dropout training
        tanhSq_a = 4. - 2. * (2. ** 0.5)
        tanhSq_b = float(-np.log(2. ** 0.5 - 1))
        
        m_out = T.tanh(m * T.inv(T.sqrt(1. + float(np.pi) * 0.5 * v)))
        v_out = 2. * (T.tanh((m - tanhSq_b * 0.5) * T.inv(T.sqrt(1. / tanhSq_a ** 2. + float(np.pi) * v * 0.5))) + 1.) \
            - T.sqr((T.tanh(m * T.inv(T.sqrt(1. + float(np.pi) * v * 0.5)))) + 1) + eps
        return m_out, v_out
