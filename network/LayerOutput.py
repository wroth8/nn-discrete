'''
The output layer.
'''

from abc import abstractmethod
from Layer import Layer

import theano
import theano.tensor as T

class LayerOutput(Layer):
    def __init__(self,
                 input_layer,
                 C,
                 objective='crossentropy',
                 enable_reparameterization_trick=False,
                 epsilon_reparameterization=1e-8,
                 srng=None):
        super(LayerOutput, self).__init__(input_layer, 'output')
        
        assert input_layer.isOutputFeatureMap() == False
        # Reparameterization trick can only be used if the input to that layer is a distribution
        assert not enable_reparameterization_trick or input_layer.isOutputDistribution()
        assert not enable_reparameterization_trick or srng is not None

        self._C = C
        self._objective = objective
        self._enable_reparameterization_trick = enable_reparameterization_trick
        self._epsilon_reparameterization = epsilon_reparameterization
        self._srng = srng
        self.t = T.ivector('t')
        
        assert objective in ['crossentropy', 'squaredhinge']

    def getTrainOutput(self):
        return self._input_layer.getTrainOutput()
    
    def getPredictionOutput(self):
        return self._input_layer.getPredictionOutput()
    
    def getSampleOutput(self):
        return self._input_layer.getSampleOutput()
    
#     def getTrainUpdates(self):
#         return self._input_layer.getTrainUpdates()
        
#     def isOutputDistribution(self):
#         return self._input_layer.isOutputDistribution()
    
    def isOutputFeatureMap(self):
        return False
    
    def getOutputType(self):
        return None
    
    def getOutputShape(self):
        return (self._C,)
    
#     def getParameters(self):
#         return self._input_layer.getParameters()
    
#     def getLayerParameters(self):
#         return []
    
    def getCost(self):
        return self.getCostLikelihood() + self.getCostRegularizer()
    
    def getCostLikelihood(self):
        if self._objective == 'crossentropy':
            if self.isOutputDistribution():
                if self._enable_reparameterization_trick:
                    a_mean, a_var = self._input_layer.getTrainOutput()
                    a_eps = self._srng.normal(a_mean.shape, 0., 1., dtype=theano.config.floatX)
                    a_sample = a_mean + a_eps * T.sqrt(a_var + self._epsilon_reparameterization)
                    a_sample = a_sample - T.max(a_sample, axis=1, keepdims=True)
                    cost = -T.sum(a_sample[T.arange(self.t.shape[0], dtype='int32'), self.t]) + T.sum(T.log(T.sum(T.exp(a_sample), axis=1)))
                    cost = cost / T.cast(self.t.shape[0], theano.config.floatX)
                else:
                    a_mean, a_var = self._input_layer.getTrainOutput()
                    a_mean = a_mean - T.max(a_mean, axis=1, keepdims=True)
    
                    a_softmax = T.exp(a_mean)
                    a_softmax = a_softmax / T.sum(a_softmax, axis=1, keepdims=True)
                
                    cost1 = -T.sum(a_mean[T.arange(self.t.shape[0], dtype='int32'), self.t]) + T.sum(T.log(T.sum(T.exp(a_mean), axis=1)))
                    cost2 = 0.5 * T.sum((a_softmax) * (1. - a_softmax) * a_var) / self.t.shape[0]
                    cost = (cost1 + cost2) / self.t.shape[0]
            else:
                a = self._input_layer.getTrainOutput()
                a = a - T.max(a, axis=1, keepdims=True)
                cost = -T.sum(a[T.arange(self.t.shape[0], dtype='int32'), self.t]) + T.sum(T.log(T.sum(T.exp(a), axis=1)))
                cost = cost / self.t.shape[0]
        elif self._objective == 'squaredhinge':
            # Make a kind of one-hot-encoding for the targets where the true
            # class is +1 and all other classes are -1. Compute the hinge
            # loss wrt to the activation in the last layer rather than the
            # softmax output
            t_one_hot = T.zeros((self.t.shape[0], self._C), dtype=theano.config.floatX)
            t_one_hot = T.set_subtensor(t_one_hot[T.arange(t_one_hot.shape[0]), self.t], 1.)
            t_one_hot = t_one_hot * 2. - 1.
            if self.isOutputDistribution():
                if self._enable_reparameterization_trick:
                    a_mean, a_var = self._input_layer.getTrainOutput()
                    a_eps = self._srng.normal(a_mean.shape, 0., 1., dtype=theano.config.floatX)
                    a_sample = a_mean + a_eps * T.sqrt(a_var + self._epsilon_reparameterization)
                    cost = T.mean(T.sqr(T.maximum(1. - t_one_hot * a_sample, 0.)))
                else:
                    return NotImplementedError()
            else:
                a = self._input_layer.getTrainOutput()
                cost = T.mean(T.sqr(T.maximum(1. - t_one_hot * a, 0.)))
        else:
            raise NotImplementedError()
        return cost
    
    def getCostRegularizer(self):
        return self._input_layer.getCost()
    
    def getSymbolicTarget(self):
        return self.t
    
    def getMessage(self):
        return self._input_layer.getMessage() + '\n%20s: OutputShape=%15s, Objective=%s' % ('LayerOutput', self.getOutputShape(), self._objective)
    
    # specific methods
    def getClassificationError(self):
        a = self._input_layer.getPredictionOutput()
        y = T.argmax(a, axis=1)
        return T.mean(T.neq(self.t, y))
    
    def getTrainClassificationError(self):
        # Computation for the true classification error computed with
        # getPredictionOutput() might follow a different path in the computation
        # graph. During training the path of getTrainOutput() is evaluated and
        # thus evaluation of the training error using getTrainOutput() could
        # reduce computation time substantially while providing a sufficient
        # approximation.
        if self.isOutputDistribution():
            a, _ = self._input_layer.getTrainOutput()
        else:
            a = self._input_layer.getTrainOutput()
        y = T.argmax(a, axis=1)
        return T.mean(T.neq(self.t, y))

    def getTrainClassificationCriterion(self):
        # Sometimes the CE is not enough and we want to compute other criterions
        # like the Top-5 error. In these cases, it is more convenient to have
        # access to the output activation directly. This function also removes
        # the need to check whether getTrainOutput returns a distribution or
        # single values.
        if self.isOutputDistribution():
            a, _ = self._input_layer.getTrainOutput()
        else:
            a = self._input_layer.getTrainOutput()
        return a