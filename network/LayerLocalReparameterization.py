'''
Performs the local reparameterization trick.
'''

from abc import abstractmethod
from Layer import Layer

import theano
import theano.tensor as T
import numpy as np
import math

class LayerLocalReparameterization(Layer):
    def __init__(self,
                 input_layer,
                 srng,
                 epsilon_reparameterization=1e-8,
                 enable_gumbel_softmax_if_binary=True,
                 gumbel_softmax_temperature=1.):
        super(LayerLocalReparameterization, self).__init__(input_layer, 'local_reparameterize')
        
        # Input layer must provide a distribution
        assert input_layer.isOutputDistribution()
        self._srng = srng
        self._epsilon_reparameterization = epsilon_reparameterization
        self._enable_gumbel_softmax_if_binary = enable_gumbel_softmax_if_binary
        self._gumbel_softmax_temperature = gumbel_softmax_temperature
        self._use_gumbel_softmax = (self._input_layer.getOutputType() in ['binary', 'binary01'] and self._enable_gumbel_softmax_if_binary)

    def getTrainOutput(self):
        x_in_mean, x_in_var = self._input_layer.getTrainOutput()
        if self._input_layer.getOutputType() == 'binary' and self._enable_gumbel_softmax_if_binary:
            # Some notes about the following lines:
            # It appears that the Theano optimizer merges 1 + eps which is numerically
            # 1. Therefore, we set eps to a minimum value of 1e-6, otherwise we get NaNs
            # if 1+erf is numerically 0.
            # Factor 0.5 can be neglected as a constant
            aux = T.stack([-x_in_mean, x_in_mean], axis=0)
            logits = T.log((1. + aux) + max(1e-6, self._epsilon_reparameterization))
            U = self._srng.uniform(logits.shape, dtype=theano.config.floatX)
            gumbel_epsilon = -T.log(-T.log(U + self._epsilon_reparameterization) + self._epsilon_reparameterization)
            logits_sample = logits + gumbel_epsilon
            logits_delta = (logits_sample[1,...] - logits_sample[0,...]) * (1. / self._gumbel_softmax_temperature)
            x_out = T.tanh(0.5 * logits_delta)
        elif self._input_layer.getOutputType() == 'binary01' and self._enable_gumbel_softmax_if_binary:
            aux = T.stack([1.-x_in_mean, x_in_mean], axis=0)
            logits = T.log(aux + max(1e-6, self._epsilon_reparameterization))
            U = self._srng.uniform(logits.shape, dtype=theano.config.floatX)
            gumbel_epsilon = -T.log(-T.log(U + self._epsilon_reparameterization) + self._epsilon_reparameterization)
            logits_sample = (logits + gumbel_epsilon) * (1. / self._gumbel_softmax_temperature)
            x_out = T.nnet.sigmoid(logits_sample[1,...] - logits_sample[0,...])
        else:
            x_eps = self._srng.normal(x_in_mean.shape, 0., 1., dtype=theano.config.floatX)
            x_out = x_in_mean + T.sqrt(x_in_var + self._epsilon_reparameterization) * x_eps
        return x_out
    
    def getPredictionOutput(self):
        return self._input_layer.getPredictionOutput()
    
    def getSampleOutput(self):
        return self._input_layer.getSampleOutput()
    
#     def getTrainUpdates(self):
#         return self._input_layer.getTrainUpdates()
    
    def isOutputDistribution(self):
        return False
    
    def getOutputType(self):
        # Note that even if the input layer is binary the gumbel softmax trick
        # gives a real-valued relaxation.
        return 'real'
    
    def getMessage(self):
        return self._input_layer.getMessage() + '\n%20s: OutputShape=%15s, DistributionOutput=%d, UseGumbelSoftmax=%d' % ('LayerLocalReparam', str(self.getOutputShape()), self.isOutputDistribution(), self._use_gumbel_softmax)
