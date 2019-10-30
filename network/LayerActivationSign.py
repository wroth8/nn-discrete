'''
Computes the sign activation function.
'''

from Layer import Layer

import theano.tensor as T

class LayerActivationSign(Layer):
    def __init__(self,
                 input_layer):
        super(LayerActivationSign, self).__init__(input_layer, 'activation_sign')
        assert input_layer.isOutputDistribution()

    def getTrainOutput(self):
        x_in_mean, x_in_var = self._input_layer.getTrainOutput()
        return LayerActivationSign.signOfGaussian(x_in_mean, x_in_var)
    
    def getPredictionOutput(self):
        x_in = self._input_layer.getPredictionOutput()
        return T.sgn(x_in)
    
    def getSampleOutput(self):
        x_in = self._input_layer.getSampleOutput()
        return T.sgn(x_in)
    
    def isOutputDistribution(self):
        return True
    
    def getOutputType(self):
        return 'binary'

    def getMessage(self):
        return self._input_layer.getMessage() + '\n%20s: OutputShape=%15s, DistributionOutput=%d' % ('LayerActivationSign', str(self.getOutputShape()), self.isOutputDistribution())
    
    # private methods section   
    
    @staticmethod
    def signOfGaussian(m, v, eps=1e-8):
        m_out = T.erf(m / T.sqrt(2. * v + eps))
        v_out = 1. - T.sqr(m_out) + max(1e-6, eps)
        return m_out, v_out
