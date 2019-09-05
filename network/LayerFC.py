'''
Fully-connected weight layer.
'''

from Layer import Layer

import theano
import theano.tensor as T
import numpy as np
import math

from network.LayerLinearForward import LayerLinearForward

class LayerFC(LayerLinearForward):
    def __init__(self,
                 input_layer,
                 n_neurons,
                 rng,
                 srng,
                 weight_type='ternaryShayer',
                 weight_parameterization=None,
                 weight_initialization_method=None,
                 enable_scale_factors=False,
                 regularizer=None,
                 regularizer_weight=None,
                 regularizer_parameters=None,
                 enable_bias=False,
                 bias_type=None,
                 bias_parameterization=None,
                 bias_initialization_method=None,
                 bias_regularizer=None,
                 bias_regularizer_weight=None,
                 bias_regularizer_parameters=None,
                 enable_reparameterization_trick=False,
                 enable_activation_normalization=False,
                 logit_bounds=None,
                 initial_parameters=None):
        super(LayerFC, self).__init__(input_layer, 'fc', logit_bounds=logit_bounds)
        # If the bias is enabled then its type must ge given
        assert not enable_bias or bias_type is not None
        # If W is real, then the bias (if enabled) must also be real
        assert not enable_bias or (weight_type != 'real' or bias_type == 'real')
        # If W is real, it does not make sense to use the reparameterization trick
        assert weight_type != 'real' or not enable_reparameterization_trick

        # For backwards compatibility
        if weight_type == 'ternaryShayer':
            print 'Warning: Weight type \'ternaryShayer\' deprecated. Use parameter \'weight_parameterization\' instead.'
            if weight_parameterization is not None or weight_initialization_method is not None:
                raise Exception()
            weight_type = 'ternary'
            weight_parameterization = 'shayer'
            weight_initialization_method = 'shayer'
        elif weight_type == 'quinaryShayer':
            print 'Warning: Weight type \'quinaryShayer\' deprecated. Use parameter \'weight_parameterization\' instead.'
            if weight_parameterization is not None or weight_initialization_method is not None:
                raise Exception()
            weight_type = 'quinary'
            weight_parameterization = 'shayer'
            weight_initialization_method = 'probability'

        self._n_neurons = n_neurons
        self._weight_type = weight_type
        self._weight_parameterization = weight_parameterization
        self._weight_initialization_method = weight_initialization_method
        self._enable_scale_factors = enable_scale_factors
        self._regularizer = regularizer
        self._regularizer_weight = regularizer_weight
        self._regularizer_parameters = regularizer_parameters # in case the regularizer has some parameters of its own
        self._enable_bias = enable_bias
        self._bias_type = bias_type
        self._bias_parameterization = bias_parameterization
        self._bias_initialization_method = bias_initialization_method
        self._bias_regularizer = bias_regularizer
        self._bias_regularizer_weight = bias_regularizer_weight
        self._bias_regularizer_parameters = bias_regularizer_parameters
        self._enable_reparameterization_trick = enable_reparameterization_trick
        self._sampling_updates = []
        self._enable_activation_normalization = enable_activation_normalization
        self._cost_regularizer = 0

        if self._input_layer.isOutputFeatureMap():
            shape_in = input_layer.getOutputShape()
            self._W_shape = (shape_in[0] * shape_in[1] * shape_in[2], self._n_neurons)
        else:
            self._W_shape = (input_layer.getOutputShape()[0], self._n_neurons)
        self._fan_in = float(self._W_shape[0] + (1 if self._enable_bias else 0))
        
        q_distributions = {
            'real'                         : lambda shape, parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng : self._addRealWeights(shape, parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng),
            'gauss'                        : lambda shape, parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng : self._addGaussDistribution(shape, parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng),
            'ternary'                      : lambda shape, parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng : self._addTernaryDistribution(shape, parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng),
            'quaternary_symmetric'         : lambda shape, parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng : self._addQuaternaryDistribution(shape, 'symmetric', parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng),
            'quaternary_fixed_point_plus'  : lambda shape, parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng : self._addQuaternaryDistribution(shape, 'fixed_point_plus', parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng),
            'quaternary_fixed_point_minus' : lambda shape, parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng : self._addQuaternaryDistribution(shape, 'fixed_point_minus', parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng),
            'quinary'                      : lambda shape, parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng : self._addQuinaryDistribution(shape, parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng)
            }

        # TODO: Maybe also implement scale factors for bias        
        self._W_mean, self._W_var, self._W_map, self._W_sample, self._W_sample_reparam, cost_regularizer = q_distributions[weight_type](self._W_shape, weight_parameterization, weight_initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng)
        self._cost_regularizer += cost_regularizer
        if self._enable_bias:
            self._b_mean, self._b_var, self._b_map, self._b_sample, self._b_sample_reparam, cost_regularizer = q_distributions[bias_type]((self._W_shape[1],), bias_parameterization, bias_initialization_method, initial_parameters, bias_regularizer, bias_regularizer_weight, bias_regularizer_parameters, False, rng, srng)
            self._cost_regularizer += cost_regularizer

    def getTrainOutput(self):
        if self._weight_type == 'real':
            # Weights are single values (not distributions)
            if self._input_layer.isOutputDistribution():
                #x_in_mean, x_in_var = self._input_layer.getTrainOutput()
                x_in_mean, x_in_var = self._getFlattenedTrainOutput()
                x_out_mean = T.dot(x_in_mean, self._W_mean)
                x_out_var = T.dot(x_in_var, T.sqr(self._W_mean))
                if self._enable_bias:
                    x_out_mean = x_out_mean + self._b_mean
            else:
                #x_in = self._input_layer.getTrainOutput()
                x_in = self._getFlattenedTrainOutput()
                x_out = T.dot(x_in, self._W_mean)
                if self._enable_bias:
                    x_out = x_out + self._b_mean
        elif self._enable_reparameterization_trick:
            # Weights are a distribution but we use the reparameterization trick
            if self._input_layer.isOutputDistribution():
                #x_in_mean, x_in_var = self._input_layer.getTrainOutput()
                x_in_mean, x_in_var = self._getFlattenedTrainOutput()
                x_out_mean = T.dot(x_in_mean, self._W_sample_reparam)
                x_out_var = T.dot(x_in_var, T.sqr(self._W_sample_reparam))
                if self._enable_bias:
                    x_out_mean = x_out_mean + self._b_sample_reparam
            else:
                #x_in = self._input_layer.getTrainOutput()
                x_in = self._getFlattenedTrainOutput()
                x_out = T.dot(x_in, self._W_sample_reparam)
                if self._enable_bias:
                    x_out = x_out + self._b_sample_reparam
        else:
            # Weights are distributions and we perform a probabilistic forward pass
            if self._input_layer.isOutputDistribution():
                #x_in_mean, x_in_var = self._input_layer.getTrainOutput()
                x_in_mean, x_in_var = self._getFlattenedTrainOutput()
                x_out_mean = T.dot(x_in_mean, self._W_mean)
                x_out_var = T.dot(T.sqr(x_in_mean), self._W_var) + \
                            T.dot(x_in_var, T.sqr(self._W_mean)) + \
                            T.dot(x_in_var, self._W_var)
            else:
                #x_in = self._input_layer.getTrainOutput()
                x_in = self._getFlattenedTrainOutput()
                x_out_mean = T.dot(x_in, self._W_mean)
                x_out_var = T.dot(T.sqr(x_in), self._W_var)
    
            if self._enable_bias:
                x_out_mean = x_out_mean + self._b_mean
                x_out_var = x_out_var + self._b_var

        if self._enable_activation_normalization:
            if self.isOutputDistribution():
                x_out_mean = x_out_mean / (self._fan_in ** 0.5)
                x_out_var = x_out_var / self._fan_in
            else:
                x_out = x_out / (self._fan_in ** 0.5)
            
        if self.isOutputDistribution():
            return x_out_mean, x_out_var
        else:
            return x_out
    
    def getPredictionOutput(self):
        #x_in = self._input_layer.getPredictionOutput()
        x_in = self._getFlattenedPredictionOutput()
        x_out = T.dot(x_in, self._W_map)
        if self._enable_bias:
            x_out = x_out + self._b_map
        if self._enable_activation_normalization:
            x_out = x_out / (self._fan_in ** 0.5)
        return x_out
    
    def getSampleOutput(self):
        #x_in = self._input_layer.getSampleOutput()
        x_in = self._getFlattenedSampleOutput()
        x_out = T.dot(x_in, self._W_sample)
        if self._enable_bias:
            x_out = x_out + self._b_sample
        if self._enable_activation_normalization:
            x_out = x_out / (self._fan_in ** 0.5)
        return x_out
    
    def getTrainUpdates(self):
        return self._input_layer.getTrainUpdates()
    
    def isOutputDistribution(self):
        if self._weight_type == 'real':
            # We use deterministic weights: The output type is the same as the input type
            return self._input_layer.isOutputDistribution()
        else:
            # We use probalistic weights: The output is deterministic if the input is
            # deterministic and the reparameterization trick is used, otherwise it is
            # a distribution.
            return self._input_layer.isOutputDistribution() or not self._enable_reparameterization_trick
    
    def isOutputFeatureMap(self):
        return False
    
    def getOutputType(self):
        return 'real'
    
    def getCost(self):
        return self._input_layer.getCost() + self._cost_regularizer
    
    def getOutputShape(self):
        return (self._W_shape[1],)
    
    def getMessage(self):
        param_names = [(p['name'], p['param'].get_value().shape) for p in self._parameter_entries]
        return self._input_layer.getMessage() + '\n%20s: OutputShape=%15s, DistributionOutput=%d, Parameters=%s, WeightType=%s, Parameterization=%s, InitMethod=%s, RegularizerType=%s, RegularizerWeight=%s, RegularizerParameter=%s' % ('LayerFC', str(self.getOutputShape()), self.isOutputDistribution(), param_names, self._weight_type, self._weight_parameterization, self._weight_initialization_method, str(self._regularizer), str(self._regularizer_weight), str(self._regularizer_parameters))

    def _getFlattenedTrainOutput(self):
        # Flattens the output if required
        if self._input_layer.isOutputFeatureMap():
            if self._input_layer.isOutputDistribution():
                x_in_mean, x_in_var = self._input_layer.getTrainOutput()
                x_in_mean = T.reshape(x_in_mean, (x_in_mean.shape[0], self._W_shape[0]))
                x_in_var = T.reshape(x_in_var, (x_in_var.shape[0], self._W_shape[0]))
                return x_in_mean, x_in_var
            else:
                x_in = self._input_layer.getTrainOutput()
                x_in = T.reshape(x_in, (x_in.shape[0], self._W_shape[0]))
                return x_in
        else:
            return self._input_layer.getTrainOutput()

    def _getFlattenedPredictionOutput(self):
        # Flattens the output if required
        if self._input_layer.isOutputFeatureMap():
            x_in = self._input_layer.getPredictionOutput()
            x_in = T.reshape(x_in, (x_in.shape[0], self._W_shape[0]))
            return x_in
        else:
            return self._input_layer.getPredictionOutput()

    def _getFlattenedSampleOutput(self):
        # Flattens the output if required
        if self._input_layer.isOutputFeatureMap():
            x_in = self._input_layer.getSampleOutput()
            x_in = T.reshape(x_in, (x_in.shape[0], self._W_shape[0]))
            return x_in
        else:
            return self._input_layer.getSampleOutput()
