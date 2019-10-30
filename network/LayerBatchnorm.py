'''
Performs batch normalization.
'''

from Layer import Layer

import theano
import theano.tensor as T
import numpy as np
import math

class LayerBatchnorm(Layer):
    def __init__(self,
                 input_layer,
                 alpha=0.01,
                 enable_shift=True,
                 enable_scale=True,
                 average_statistics_over_predictions=False,
                 initial_parameters=None):
        super(LayerBatchnorm, self).__init__(input_layer, 'batchnorm')
        
        # At least either shift or scale must be enabled
        assert enable_shift or enable_scale
        self._alpha = alpha
        self._enable_shift = enable_shift
        self._enable_scale = enable_scale
        self._average_statistics_over_predictions = average_statistics_over_predictions
        self._epsilon = 1e-8
        
        # TODO: allow prediction statistics to be in the initial parameters
        self._bn_updates = []
        self._bn_updates_prediction = [] # updates for batchnorm statistics according to prediction path, i.e., the path without sampling and variances
        self._bn_updates_swap = [] # swaps the running batchnorm statistics for the training path and the prediction path
        x_in_prediction = self._input_layer.getPredictionOutput()

        if self._input_layer.isOutputDistribution():
            self._x_in_mean, self._x_in_var = self._input_layer.getTrainOutput()
            if self._input_layer.isOutputFeatureMap():
                self._batch_mean = T.mean(self._x_in_mean, axis=[0,2,3])
                self._batch_inv_std = T.inv(T.sqrt(T.sum(self._x_in_var + T.sqr(self._x_in_mean - self._batch_mean[:,None,None]), axis=[0,2,3]) / (T.cast(self._x_in_mean.shape[0] * self._x_in_mean.shape[2] * self._x_in_mean.shape[3], theano.config.floatX) - 1.) + self._epsilon))
            else:
                self._batch_mean = T.mean(self._x_in_mean, axis=0)
                self._batch_inv_std = T.inv(T.sqrt(T.sum(self._x_in_var + T.sqr(self._x_in_mean - self._batch_mean), axis=0) / (T.cast(self._x_in_mean.shape[0], theano.config.floatX) - 1.) + self._epsilon))
        else:
            if self._input_layer.isOutputFeatureMap():
                self._x_in = self._input_layer.getTrainOutput()
                self._batch_mean = T.mean(self._x_in, axis=[0,2,3])
                self._batch_inv_std = T.inv(T.sqrt(T.var(self._x_in, axis=[0,2,3]) + self._epsilon))
            else:
                self._x_in = self._input_layer.getTrainOutput()
                self._batch_mean = T.mean(self._x_in, axis=0)
                self._batch_inv_std = T.inv(T.sqrt(T.var(self._x_in, axis=0) + self._epsilon))

        if self._input_layer.isOutputFeatureMap():
            self._batch_mean_prediction = T.mean(x_in_prediction, axis=[0,2,3])
            self._batch_inv_std_prediction = T.inv(T.sqrt(T.sum(T.sqr(x_in_prediction - self._batch_mean_prediction[:,None,None]), axis=[0,2,3]) / (T.cast(x_in_prediction.shape[0] * x_in_prediction.shape[2] * x_in_prediction.shape[3], theano.config.floatX) - 1.) + self._epsilon))
        else:
            self._batch_mean_prediction = T.mean(x_in_prediction, axis=0)
            self._batch_inv_std_prediction = T.inv(T.sqrt(T.sum(T.sqr(x_in_prediction - self._batch_mean_prediction), axis=0) / (T.cast(x_in_prediction.shape[0], theano.config.floatX) - 1.) + self._epsilon))
        
        if self._enable_shift:
            avg_batch_mean_prediction_values = np.zeros((self._input_layer.getOutputShape()[0],), theano.config.floatX) # TODO: this could be made part of the initial parameters
            if initial_parameters is None:
                beta_values = np.zeros((self._input_layer.getOutputShape()[0],), theano.config.floatX)
                avg_batch_mean_values = np.zeros((self._input_layer.getOutputShape()[0],), theano.config.floatX)
            else:
                if 'beta' not in initial_parameters:
                    raise Exception('Initialization parameter \'beta\' not found')
                if 'avg_batch_mean' not in initial_parameters:
                    raise Exception('Initialization parameter \'avg_batch_mean\' not found')
                if initial_parameters['beta'].shape != (self._input_layer.getOutputShape()[0],):
                    raise Exception('Initialization parameter \'beta\' must have shape (%s) but has shape (%s)'
                        % (str(self._input_layer.getOutputShape()[0]), ','.join(map(str, initial_parameters['beta'].shape))))
                if initial_parameters['avg_batch_mean'].shape != (self._input_layer.getOutputShape()[0],):
                    raise Exception('Initialization parameter \'avg_batch_mean\' must have shape (%s) but has shape (%s)'
                        % (str(self._input_layer.getOutputShape()[0]), ','.join(map(str, initial_parameters['avg_batch_mean'].shape))))
                beta_values = initial_parameters['beta']
                avg_batch_mean_values = initial_parameters['avg_batch_mean']
            self._beta = theano.shared(beta_values, borrow=True)
            self._avg_batch_mean = theano.shared(avg_batch_mean_values, borrow=True)
            #self._avg_batch_mean_prediction = theano.shared(avg_batch_mean_prediction_values, borrow=True)
            self._addParameterEntry(self._beta, 'beta', is_trainable=True)
            self._addParameterEntry(self._avg_batch_mean, 'avg_batch_mean', is_trainable=False)
            #self._addParameterEntry(self._avg_batch_mean_prediction, 'avg_batch_mean_prediction', is_trainable=False)
            # Batch normalization running average parameter: alpha * new + (1 - alpha) * old
            if self._average_statistics_over_predictions == True:
                self._bn_updates += [(self._avg_batch_mean, self._alpha * self._batch_mean_prediction + (1. - self._alpha) * self._avg_batch_mean)]
            else:
                self._bn_updates += [(self._avg_batch_mean, self._alpha * self._batch_mean + (1. - self._alpha) * self._avg_batch_mean)]
            # TODO: Does not seem important, remove
            #self._bn_updates_prediction += [(self._avg_batch_mean_prediction, self._alpha * self._batch_mean_prediction + (1. - self._alpha) * self._avg_batch_mean_prediction)]
            #self._bn_updates_swap += [(self._avg_batch_mean, self._avg_batch_mean_prediction),
            #                          (self._avg_batch_mean_prediction, self._avg_batch_mean)]

            if self.isOutputFeatureMap():
                # The parameters (shared variables) are still only 1d
                self._beta = self._beta[:,None,None]
                self._batch_mean = self._batch_mean[:,None,None]
                self._avg_batch_mean = self._avg_batch_mean[:,None,None]
            
        if self._enable_scale:
            avg_batch_inv_std_prediction_values = np.ones((self._input_layer.getOutputShape()[0],), theano.config.floatX) # TODO: this could be made part of the initial parameters
            if initial_parameters is None:
                gamma_values = np.ones((self._input_layer.getOutputShape()[0],), theano.config.floatX)
                avg_batch_inv_std_values = np.ones((self._input_layer.getOutputShape()[0],), theano.config.floatX)
            else:
                if 'beta' not in initial_parameters:
                    raise Exception('Initialization parameter \'gamma\' not found')
                if 'avg_batch_mean' not in initial_parameters:
                    raise Exception('Initialization parameter \'avg_batch_inv_std\' not found')
                if initial_parameters['beta'].shape != (self._input_layer.getOutputShape()[0],):
                    raise Exception('Initialization parameter \'gamma\' must have shape (%s) but has shape (%s)'
                        % (str(self._input_layer.getOutputShape()[0]), ','.join(map(str, initial_parameters['gamma'].shape))))
                if initial_parameters['avg_batch_inv_std'].shape != (self._input_layer.getOutputShape()[0],):
                    raise Exception('Initialization parameter \'avg_batch_inv_std\' must have shape (%s) but has shape (%s)' 
                        % (str(self._input_layer.getOutputShape()[0]), ','.join(map(str, initial_parameters['avg_batch_inv_std'].shape))))
                gamma_values = initial_parameters['gamma']
                avg_batch_inv_std_values = initial_parameters['avg_batch_inv_std']

            self._gamma = theano.shared(gamma_values, borrow=True)
            self._avg_batch_inv_std = theano.shared(avg_batch_inv_std_values, borrow=True)
            ###self._avg_batch_inv_std_prediction = theano.shared(avg_batch_inv_std_prediction_values, borrow=True) # TODO: Does not seem important, remove
            self._addParameterEntry(self._gamma, 'gamma', is_trainable=True)
            self._addParameterEntry(self._avg_batch_inv_std, 'avg_batch_inv_std', is_trainable=False)
            ###self._addParameterEntry(self._avg_batch_inv_std_prediction, 'avg_batch_inv_std_prediction', is_trainable=False) # TODO: Does not seem important, remove
            # Batch normalization running average parameter: alpha * new + (1 - alpha) * old
            if self._average_statistics_over_predictions == True:
                self._bn_updates += [(self._avg_batch_inv_std, self._alpha * self._batch_inv_std_prediction + (1. - self._alpha) * self._avg_batch_inv_std)]
            else:
                self._bn_updates += [(self._avg_batch_inv_std, self._alpha * self._batch_inv_std + (1. - self._alpha) * self._avg_batch_inv_std)]
            # TODO: Does not seem important, remove
            #self._bn_updates_prediction += [(self._avg_batch_inv_std_prediction, self._alpha * self._batch_inv_std_prediction + (1. - self._alpha) * self._avg_batch_inv_std_prediction)]
            #self._bn_updates_swap += [(self._avg_batch_inv_std, self._avg_batch_inv_std_prediction),
            #                          (self._avg_batch_inv_std_prediction, self._avg_batch_inv_std)]

            if self.isOutputFeatureMap():
                # The parameters (shared variables) are still only 1d
                self._gamma = self._gamma[:,None,None]
                self._batch_inv_std = self._batch_inv_std[:,None,None]
                self._avg_batch_inv_std = self._avg_batch_inv_std[:,None,None]

            self._a_train = self._batch_inv_std * self._gamma
            self._b_train = self._beta - self._batch_inv_std * self._gamma * self._batch_mean
            self._a_predict = self._avg_batch_inv_std * self._gamma
            self._b_predict = self._beta - self._avg_batch_inv_std * self._gamma * self._avg_batch_mean

    def getTrainOutput(self):
        if self._input_layer.isOutputDistribution():
            # Stochastic batch normalization implemented according to
            # Probabilistic Binary Neural Networks
            # J.W.T. Peters and M. Welling
            # https://arxiv.org/abs/1809.03368
            # arXiv version 10 Sep 2018
            x_in_mean, x_in_var = self._x_in_mean, self._x_in_var
            
            #if self._enable_shift:
            #    x_in_mean = x_in_mean - self._batch_mean
            #if self._enable_scale:
            #    x_in_mean = x_in_mean * self._batch_inv_std
            #    x_in_mean = x_in_mean * self._gamma
            #    x_in_var = x_in_var * T.sqr(self._batch_inv_std)
            #    x_in_var = x_in_var * T.sqr(self._gamma)
            #if self._enable_shift:
            #    x_in_mean = x_in_mean + self._beta
            #x_out_mean, x_out_var = x_in_mean, x_in_var
            x_out_mean = x_in_mean * self._a_train + self._b_train
            x_out_var = x_in_var * T.sqr(self._a_train)
            
            return x_out_mean, x_out_var
        else:
            x_in = self._x_in

            #if self._enable_shift:
            #    x_in = x_in - self._batch_mean
            #if self._enable_scale:
            #    x_in = x_in * self._batch_inv_std
            #    x_in = x_in * self._gamma
            #if self._enable_shift:
            #    x_in = x_in + self._beta
            #x_out = x_in
            x_out = x_in * self._a_train + self._b_train

            return x_out
    
    def getPredictionOutput(self):
        x_in = self._input_layer.getPredictionOutput()
        #if self._enable_shift:
        #    x_in = x_in - self._avg_batch_mean
        #if self._enable_scale:
        #    x_in = x_in * self._avg_batch_inv_std
        #    x_in = x_in * self._gamma
        #if self._enable_shift:
        #    x_in = x_in + self._beta
        #x_out = x_in
        x_out = x_in * self._a_predict + self._b_predict
        return x_out
    
    def getSampleOutput(self):
        x_in = self._input_layer.getSampleOutput()
        #if self._enable_shift:
        #    x_in = x_in - self._avg_batch_mean
        #if self._enable_scale:
        #    x_in = x_in * self._avg_batch_inv_std
        #    x_in = x_in * self._gamma
        #if self._enable_shift:
        #    x_in = x_in + self._beta
        #x_out = x_in
        x_out = x_in * self._a_predict + self._b_predict
        return x_out
    
    def getTrainUpdates(self):
        return self._input_layer.getTrainUpdates() + self._bn_updates

    def getLayerSpecificValues(self, layer_type):
        res = self._input_layer.getLayerSpecificValues(layer_type)
        if layer_type == self._layer_type:
            # updates for batchnorm statistics according to the prediction path
            if 'bn_updates_prediction' not in res:
                res['bn_updates_prediction'] = []
            res['bn_updates_prediction'] += self._bn_updates_prediction
            # updates that swap the batchnorm statistics of the train and the prediction path
            if 'bn_updates_swap' not in res:
                res['bn_updates_swap'] = []
            res['bn_updates_swap'] += self._bn_updates_swap
        return res
    
    def getOutputType(self):
        return 'real'
    
    def getMessage(self):
        param_names = [(p['name'], p['param'].get_value().shape) for p in self._parameter_entries]
        return self._input_layer.getMessage() + '\n%20s: OutputShape=%15s, DistributionOutput=%d, Parameters=%s' % ('LayerBatchnorm', str(self.getOutputShape()), self.isOutputDistribution(), param_names)
