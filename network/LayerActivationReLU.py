'''
Computes the ReLU activation function.
'''

from abc import abstractmethod
from Layer import Layer

import theano
import theano.tensor as T
import numpy as np
import math

class LayerActivationRelu(Layer):
    def __init__(self,
                 input_layer):
        super(LayerActivationRelu, self).__init__(input_layer, 'activation_relu')

    def getTrainOutput(self):
        if self._input_layer.isOutputDistribution():
            x_in_mean, x_in_var = self._input_layer.getTrainOutput()
            return LayerActivationRelu.reluOfGaussian(x_in_mean, x_in_var)
        else:
            x_in = self._input_layer.getTrainOutput()
            x_out = T.nnet.relu(x_in)
            return x_out
    
    def getPredictionOutput(self):
        x_in = self._input_layer.getPredictionOutput()
        return T.nnet.relu(x_in)
    
    def getSampleOutput(self):
        x_in = self._input_layer.getSampleOutput()
        return T.nnet.relu(x_in)
    
    def getOutputType(self):
        return 'real'
    
    def getMessage(self):
        return self._input_layer.getMessage() + '\n%20s: OutputShape=%15s, DistributionOutput=%d' % ('LayerActivationReLU', str(self.getOutputShape()), self.isOutputDistribution())
    
    # private methods section   
    
    @staticmethod
    def reluOfGaussian(m, v, eps=1e-8):
        # Gaussian approximation of the relu applied to a Gaussian
        # Implementation according to:
        # Hernandez-Lobato and Adams; Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks
        pdf_norm = float(1. / (2. * np.pi)**0.5)
        alpha = m / T.sqrt(v + eps)
        alpha_inv = T.inv(alpha)
        alpha_div_sqrt2 = alpha * (0.5 ** 0.5)
        pdf_alpha = pdf_norm * T.exp(-0.5 * T.sqr(alpha))
          
        cdf_alpha_pos = 0.5 * (1. + T.erf(alpha_div_sqrt2))
        cdf_alpha_neg = 0.5 * (1. + T.erf(-alpha_div_sqrt2)) # TODO: try with 1. - cdf_alpha_pos
          
        gamma1 = pdf_alpha / cdf_alpha_pos
        gamma2 = -alpha - alpha_inv + 2.0 * alpha_inv ** 3.
        gamma = T.switch(T.ge(alpha, -10.), gamma1, gamma2)
          
        v_aux = m + T.sqrt(v + eps) * gamma
        m_out = cdf_alpha_pos * v_aux
        v_out = m_out * v_aux * cdf_alpha_neg + cdf_alpha_pos * v * (1. - gamma * (gamma + alpha))
        v_out = T.maximum(v_out, eps)
        return m_out, v_out

#         # Old implementation
#         a = 0 # slope of the negative part (0 --> ReLU)
#         val1 = T.sqrt(v)
#         val2 = m / T.sqrt(2) / val1
#         val3 = T.erf(val2)
#         val4 = val1 / T.sqrt(2 * math.pi) * T.exp(-val2 ** 2)
#         m1_out = 0.5 * m * (1 + a + (1 - a) * val3) + (1 - a) * val4
#         m2_out = 0.5 * (m ** 2 + v) * (1 + a ** 2 + (1 - a ** 2) * val3) + (1 - a ** 2) * m * val4
#         v_out = (m2_out - m1_out ** 2.) + eps
#         return m1_out, v_out
