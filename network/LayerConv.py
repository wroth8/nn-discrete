'''
Convolutional layer.
'''

from Layer import Layer

import theano
import theano.tensor as T
import numpy as np
import math

from network.LayerLinearForward import LayerLinearForward

class LayerConv(LayerLinearForward):
    def __init__(self,
                 input_layer,
                 n_features_out,
                 kernel_size,
                 stride,
                 border_mode,
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
                 compute_cross_correlation=False,
                 logit_bounds=None,
                 initial_parameters=None):
        super(LayerConv, self).__init__(input_layer, 'conv', logit_bounds=logit_bounds)
        # The input must not be a feature map
        assert input_layer.isOutputFeatureMap() == True
        # If the bias is enabled then its type must ge given
        assert not enable_bias or bias_type is not None
        # If W is real, then the bias (if enabled) must also be real
        assert not enable_bias or (weight_type != 'real' or bias_type == 'real')
        # If W is real, it does not make sense to use the reparameterization trick
        assert weight_type != 'real' or not enable_reparameterization_trick
        # kernel_size must be a 2-tuple
        assert isinstance(kernel_size, tuple) and len(kernel_size) == 2
        # stride must be a 2-tuple
        assert isinstance(stride, tuple) and len(stride) == 2
        # border_mode must be in ['valid', 'full', 'half', 'same']
        assert border_mode in ['valid', 'full', 'half', 'same']

        '''        
        Important notes on the border mode:
        The border modes 'valid', 'full', and 'half' are Theano implementations.
        The border mode 'same' is implemented to give the same result as the
        'same' padding from TensorFlow. However, note that our Theano implementation,
        depending on the strides and window sizes, probably uses manual padding to
        accomplish this which is probably slower than a native implementation.
        Documentation of Theano border modes ['valid', 'full', 'half']:
          - http://deeplearning.net/software/theano/library/tensor/nnet/conv.html
        Documentation for TensorFlows border mode ['same']:
          - https://www.tensorflow.org/api_guides/python/nn#Convolution
        '''
        if border_mode == 'same':
            in_shape = self._input_layer.getOutputShape()
            if (in_shape[1] % stride[0] == 0):
                pad0 = max(kernel_size[0] - stride[0], 0)
            else:
                pad0 = max(kernel_size[0] - (in_shape[1] % stride[0]), 0)
            if (in_shape[2] % stride[1] == 0):
                pad1 = max(kernel_size[1] - stride[1], 0)
            else:
                pad1 = max(kernel_size[1] - (in_shape[2] % stride[1]), 0)
            pad = ((pad0 // 2, pad0 - pad0 // 2), (pad1 // 2, pad1 - pad1 // 2))
        
            self._tf_same_padding_row = pad[0][0] != pad[0][1]
            self._tf_same_padding_col = pad[1][0] != pad[1][1]
            self._tf_same_padding = self._tf_same_padding_row or self._tf_same_padding_col
            border_mode = (pad[0][0], pad[1][0]) # The first entry is the smaller one which is needed here
            print 'Experimental: Do tf_same_padding:', self._tf_same_padding, ', border_mode changed to:', border_mode
        else:
            self._tf_same_padding, self._tf_same_padding_row, self._tf_same_padding_col = False, False, False

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

        self._n_features_out = n_features_out
        self._n_features_in = self._input_layer.getOutputShape()[0]
        self._kernel_size = kernel_size
        self._stride = stride
        self._border_mode = border_mode
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
        self._compute_cross_correlation = compute_cross_correlation
        self._cost_regularizer = 0

        self._fan_in = float(self._n_features_in * self._kernel_size[0] * self._kernel_size[1] + (1 if self._enable_bias else 0))
        self._W_shape = (self._n_features_out, self._n_features_in) + kernel_size

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
            self._b_mean, self._b_var, self._b_map, self._b_sample, self._b_sample_reparam, cost_regularizer = q_distributions[bias_type]((self._W_shape[0],), bias_parameterization, bias_initialization_method, initial_parameters, bias_regularizer, bias_regularizer_weight, bias_regularizer_parameters, False, rng, srng)
            self._cost_regularizer += cost_regularizer
            
        self._conv_args = {'border_mode' : self._border_mode,
                           'subsample'   : self._stride,
                           'filter_flip' : self._compute_cross_correlation==False}

    def _pad(self, x):
        if self._tf_same_padding == True:
            # Add zero padding manually to simulate tensorflows SAME padding
            # This cumbersome implementation is necessary sicne theano does not support asymmetric padding
            in_shape = self._input_layer.getOutputShape()
            if self._input_layer.isOutputDistribution():
                x_mean, x_var = x
                if self._tf_same_padding_row:
                    zero_pad = T.zeros((x_mean.shape[0], in_shape[0], 1, in_shape[2]), theano.config.floatX)
                    x_mean = T.concatenate([x_mean, zero_pad], axis=2)
                    x_var = T.concatenate([x_var, zero_pad], axis=2)
                if self._tf_same_padding_col:
                    zero_pad = T.zeros((x_mean.shape[0], in_shape[0], in_shape[1] + (1 if self._tf_same_padding_row else 0), 1), theano.config.floatX)
                    x_mean = T.concatenate([x_mean, zero_pad], axis=3)
                    x_var = T.concatenate([x_var, zero_pad], axis=3)
                x = (x_mean, x_var)
            else:
                if self._tf_same_padding_row:
                    zero_pad = T.zeros((x.shape[0], in_shape[0], 1, in_shape[2]), theano.config.floatX)
                    x = T.concatenate([x, zero_pad], axis=2)
                if self._tf_same_padding_col:
                    zero_pad = T.zeros((x.shape[0], in_shape[0], in_shape[1] + (1 if self._tf_same_padding_row else 0), 1), theano.config.floatX)
                    x = T.concatenate([x, zero_pad], axis=3)
        return x
        
    def getTrainOutput(self):
        if self._weight_type == 'real':
            # Weights are single values (not distributions)
            if self._input_layer.isOutputDistribution():
                x_in_mean, x_in_var = self._pad(self._input_layer.getTrainOutput())
                
                x_out_mean = T.nnet.conv2d(x_in_mean, self._W_mean, **self._conv_args)
                x_out_var = T.nnet.conv2d(x_in_var, T.sqr(self._W_mean), **self._conv_args)
                if self._enable_bias:
                    x_out_mean = x_out_mean + self._b_mean[:,None,None]
            else:
                x_in = self._pad(self._input_layer.getTrainOutput())
                x_out = T.nnet.conv2d(x_in, self._W_mean, **self._conv_args)
                if self._enable_bias:
                    x_out = x_out + self._b_mean[:,None,None]
        elif self._enable_reparameterization_trick:
            # Weights are a distribution but we use the reparameterization trick
            if self._input_layer.isOutputDistribution():
                x_in_mean, x_in_var = self._pad(self._input_layer.getTrainOutput())
                x_out_mean = T.nnet.conv2d(x_in_mean, self._W_sample_reparam, **self._conv_args)
                x_out_var = T.nnet.conv2d(x_in_var, T.sqr(self._W_sample_reparam), **self._conv_args)
                if self._enable_bias:
                    x_out_mean = x_out_mean + self._b_sample_reparam[:,None,None]
            else:
                x_in = self._pad(self._input_layer.getTrainOutput())
                x_out = T.nnet.conv2d(x_in, self._W_sample_reparam, **self._conv_args)
                if self._enable_bias:
                    x_out = x_out + self._b_sample_reparam[:,None,None]
        else:
            # Weights are distributions and we perform a probabilistic forward pass
            if self._input_layer.isOutputDistribution():
                x_in_mean, x_in_var = self._pad(self._input_layer.getTrainOutput())
                x_out_mean = T.nnet.conv2d(x_in_mean, self._W_mean, **self._conv_args)
                x_out_var = T.nnet.conv2d(T.sqr(x_in_mean), self._W_var, **self._conv_args) + \
                            T.nnet.conv2d(x_in_var, T.sqr(self._W_mean), **self._conv_args) + \
                            T.nnet.conv2d(x_in_var, self._W_var, **self._conv_args)
            else:
                x_in = self._pad(self._input_layer.getTrainOutput())
                x_out_mean = T.nnet.conv2d(x_in, self._W_mean, **self._conv_args)
                x_out_var = T.nnet.conv2d(T.sqr(x_in), self._W_var, **self._conv_args)
    
            if self._enable_bias:
                x_out_mean = x_out_mean + self._b_mean[:,None,None]
                x_out_var = x_out_var + self._b_var[:,None,None]

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
        x_in = self._pad(self._input_layer.getPredictionOutput())
        x_out = T.nnet.conv2d(x_in, self._W_map, **self._conv_args)
        if self._enable_bias:
            x_out = x_out + self._b_map[:,None,None]
        if self._enable_activation_normalization:
            x_out = x_out / (self._fan_in ** 0.5)
        return x_out
    
    def getSampleOutput(self):
        x_in = self._pad(self._input_layer.getSampleOutput())
        x_out = T.nnet.conv2d(x_in, self._W_sample, **self._conv_args)
        if self._enable_bias:
            x_out = x_out + self._b_sample[:,None,None]
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
        return True
    
    def getOutputType(self):
        return 'real'
    
    def getCost(self):
        return self._input_layer.getCost() + self._cost_regularizer
    
    def getOutputShape(self):
        # Output shapes are computed according to
        # http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
        _, rows, cols = self._input_layer.getOutputShape()
        if self._border_mode == 'valid':
            # No padding
            rows = (rows - self._kernel_size[0]) // self._stride[0] + 1
            cols = (cols - self._kernel_size[1]) // self._stride[1] + 1
        elif self._tf_same_padding:
            rows = np.ceil(float(rows) / self._stride[0]).astype(type(rows))
            cols = np.ceil(float(cols) / self._stride[1]).astype(type(cols))
        else:
            if self._border_mode == 'full':
                # Padding of size kernel_size - 1
                pad = self._kernel_size[0] - 1, self._kernel_size[1] - 1
            elif self._border_mode == 'half':
                # Padding of size kernel_size // 2
                pad = self._kernel_size[0] // 2, self._kernel_size[1] // 2
            elif isinstance(self._border_mode, tuple) and len(self._border_mode) == 2:
                # Padding is given explicitly
                pad = self._border_mode
            else:
                raise Exception('Unknown border mode \'%s\'. We should not be here (error should have been catched earlier).' % (self._border_mode))
            rows = (rows + 2 * pad[0] - self._kernel_size[0]) // self._stride[0] + 1
            cols = (cols + 2 * pad[1] - self._kernel_size[1]) // self._stride[1] + 1
        return (self._n_features_out, rows, cols)
    
    def getMessage(self):
        param_names = [(p['name'], p['param'].get_value().shape) for p in self._parameter_entries]
        return self._input_layer.getMessage() + '\n%20s: OutputShape=%15s, DistributionOutput=%d, Parameters=%s, WeightType=%s, Parameterization=%s, InitMethod=%s, KernelSize=%s, Stride=%s, RegularizerType=%s, RegularizerWeight=%s, RegularizerParameter=%s' % ('LayerConv', str(self.getOutputShape()), self.isOutputDistribution(), str(param_names), self._weight_type, self._weight_parameterization, self._weight_initialization_method, str(self._kernel_size), str(self._stride), str(self._regularizer), str(self._regularizer_weight), str(self._regularizer_parameters))
