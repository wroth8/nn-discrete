'''
Super-class for linear layers (fully-connected and convolutional layers).
Implements initialization and parameterization of weights.
'''

from Layer import Layer

import theano
import theano.tensor as T
import math
import numpy as np

from scipy.optimize import lsq_linear
from scipy.interpolate import interp1d

from abc import ABCMeta

class LayerLinearForward(Layer):
    __metaclass__ = ABCMeta
    
    def __init__(self, input_layer, layer_type, logit_bounds=None):
        super(LayerLinearForward, self).__init__(input_layer, layer_type)
        self._logit_bounds = logit_bounds
        
    def getSymbolicWeights(self, weight_types):
        def aux_get_param(param_name):
            return [t['param'] for t in self.getLayerParameterEntries() if t['name'] == param_name][0]
        symbolic_weights = ()
        for weight_type in weight_types:
            if weight_type == 'raw':
                if self._weight_type == 'real':
                    raw = (aux_get_param('W'),)
                elif self._weight_type == 'gauss':
                    raw = (aux_get_param('W_mu'), aux_get_param('W_sigma_rho'))
                elif self._weight_type in ['ternary',
                                           'quaternary_symmetric',
                                           'quaternary_fixed_point_plus',
                                           'quaternary_fixed_point_minus',
                                           'quinary']:
                    raw = (aux_get_param('W_rho'),)
                elif self._weight_type == 'ternaryShayer':
                    raw = (aux_get_param('W_rhoA'), aux_get_param('W_rhoB'))
                else:
                    raise NotImplementedError()
                symbolic_weights = symbolic_weights + raw
            elif weight_type == 'bias_raw':
                if self._weight_type == 'real':
                    raw = (aux_get_param('b'),)
                elif self._weight_type == 'gauss':
                    raw = (aux_get_param('b_mu'), aux_get_param('b_sigma_rho'))
                elif self._weight_type in ['ternary',
                                           'quaternary_symmetric',
                                           'quaternary_fixed_point_plus',
                                           'quaternary_fixed_point_minus',
                                           'quinary']:
                    raw = (aux_get_param('b_rho'),)
                elif self._weight_type == 'ternaryShayer':
                    raw = (aux_get_param('b_rhoA'), aux_get_param('b_rhoB'))
                else:
                    raise NotImplementedError()
                symbolic_weights = symbolic_weights + raw
            elif weight_type == 'mean':
                symbolic_weights = symbolic_weights + (self._W_mean,)
            elif weight_type == 'bias_mean':
                symbolic_weights = symbolic_weights + (self._b_mean,)
            elif weight_type == 'var':
                symbolic_weights = symbolic_weights + (self._W_var,)
            elif weight_type == 'bias_var':
                symbolic_weights = symbolic_weights + (self._b_var,)
            elif weight_type == 'map':
                symbolic_weights = symbolic_weights + (self._W_map,)
            elif weight_type == 'bias_map':
                symbolic_weights = symbolic_weights + (self._b_map,)
            elif weight_type == 'sample':
                symbolic_weights = symbolic_weights + (self._W_sample,)
            elif weight_type == 'bias_sample':
                symbolic_weights = symbolic_weights + (self._b_sample,)
            elif weight_type == 'scale_factor_raw':
                symbolic_weights = symbolic_weights + ([t[1] for t in self._parameters_names if t[0] == 'scale_factor_rho'][0],)
            else:
                raise NotImplementedError()
                
        return self._input_layer.getSymbolicWeights(weight_types) + [symbolic_weights]
        
    def _addRealWeights(self, shape, parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng):
        assert enable_scale_factors == False
        # prior_param determines the variance of the zero-mean Gaussian prior distribution
        is_bias = len(shape) == 1
        param_name_W = '%s' % ('b' if is_bias else 'W')
        if initial_parameters is None:
            if is_bias:
                W_values = np.zeros(shape, theano.config.floatX)
            else:
                scale = (2. / self._fan_in) ** 0.5
                #scale = (1. / self._fan_in) ** 0.5 # use this for tanh
                W_values = rng.uniform(low=-math.sqrt(3)*scale,
                                       high=math.sqrt(3)*scale,
                                       size=shape).astype(theano.config.floatX)
        else:
            W_values = LayerLinearForward._assignParameters(initial_parameters, param_name_W, shape)
        
        W = theano.shared(value=W_values, borrow=True)
        self._addParameterEntry(W, param_name_W, is_trainable=True)
        
        if regularizer == 'l2':
            assert regularizer_parameters is None
            assert regularizer_weight is not None
            cost_regularizer = T.sum(T.sqr(W)) * regularizer_weight
        elif regularizer == 'l1':
            assert regularizer_parameters is None
            assert regularizer_weight is not None
            cost_regularizer = T.sum(T.abs_(W)) * regularizer_weight            
        else:
            cost_regularizer = 0
            
        return W, T.constant(0., dtype=theano.config.floatX), W, W, W, cost_regularizer
        
    def _addGaussDistribution(self, shape, parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng):
        assert enable_scale_factors == False
        if self.isOutputFeatureMap():
            print 'Warning: We did not check convolutions for _addGaussDistribution'
        # prior_param determines the variance of the zero-mean Gaussian prior distribution
        is_bias = len(shape) == 1
        param_name = 'b' if is_bias else 'W'
        param_name_mu = '%s_mu' % (param_name)
        param_name_sigma_rho = '%s_sigma_rho' % (param_name)
        if initial_parameters is None:
            # TODO: Initialization with W/b parameters from non-Bayesian NN (as in ternary methods)
            scale = (2. / self._fan_in) ** 0.5
            W_mu_values = rng.uniform(low=-math.sqrt(3)*scale,
                                      high=math.sqrt(3)*scale,
                                      size=shape).astype(theano.config.floatX)
            W_sigma_rho_values = np.full(shape, -5., dtype=theano.config.floatX)
        else:
            if param_name_mu in initial_parameters and param_name in initial_parameters:
                raise Exception('Initial parameters contain \'%s\' and \'%s\'. Do not know which initial parameters to use.' % (param_name_mu, param_name))
            if param_name_mu in initial_parameters:
                W_mu_values = LayerLinearForward._assignParameters(initial_parameters, param_name_mu, shape)
            elif param_name in initial_parameters:
                W_mu_values = LayerLinearForward._assignParameters(initial_parameters, param_name, shape)
            else:
                raise Exception('Initialization parameter \'%s\' or \'%s\' not found' % param_name_mu, param_name)

            if param_name_sigma_rho in initial_parameters:
                W_sigma_rho_values = LayerLinearForward._assignParameters(initial_parameters, param_name_sigma_rho, shape)
            else:
                print 'Note: Initialization parameter \'%s\' not found. Using default initial values for the variance.' % (param_name_sigma_rho)
                W_sigma_rho_values = np.full(shape, -5., dtype=theano.config.floatX)
        
        W_mu = theano.shared(value=W_mu_values, borrow=True)
        W_sigma_rho = theano.shared(value=W_sigma_rho_values, borrow=True)
        self._addParameterEntry(W_mu, param_name_mu, is_trainable=True)
        self._addParameterEntry(W_sigma_rho, param_name_sigma_rho, is_trainable=True)

        W_mean = W_mu
        W_var = T.nnet.softplus(W_sigma_rho)
        W_map = W_mu

        W_sample_values = np.zeros(shape, dtype=theano.config.floatX)
        W_sample = theano.shared(W_sample_values, borrow=True)
        W_epsilon = srng.normal(shape, dtype=theano.config.floatX)
        W_resampled = W_mu + W_epsilon * T.sqrt(W_var)
        self._sampling_updates += [(W_sample, W_resampled)]
        print 'Warning: Sampling for distribution \'real\' was not tested'
    
        W_sample_reparam = W_resampled
        
        if regularizer == 'kl':
            assert regularizer_parameters is not None
            assert regularizer_weight is not None
            log_2_pi = float(np.log(2. * np.pi))
            log_2_pi_gamma = float(np.log(2. * np.pi * regularizer_parameters))
            cost_entropy = -0.5 * ((1. + log_2_pi) * np.prod(shape) + T.sum(T.log(W_var)))
            cost_prior = 0.5 * ( log_2_pi_gamma * np.prod(shape) + (1. / regularizer_parameters) * (T.sum(T.sqr(W_mu)) + T.sum(W_var)) )
            cost_regularizer = (cost_entropy + cost_prior) * regularizer_weight
        else:
            cost_regularizer = 0
            
        return W_mean, W_var, W_map, W_sample, W_sample_reparam, cost_regularizer
    
    def _addTernaryDistribution(self, shape, parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng):
        if parameterization is None:
            parameterization = 'shayer'
        if initialization_method is None:
            initialization_method = 'shayer'
            if enable_scale_factors:
                initialization_method_scale_factors = 'default'
        elif enable_scale_factors:
            if isinstance(initialization_method, tuple):
                assert len(initialization_method) == 2
                initialization_method, initialization_method_scale_factors = \
                  initialization_method[0], initialization_method[1]
            else:
                initialization_method_scale_factors = 'default'
  
        # prior_param determines the prior probability p(w=0). We have p(w=1)=p(w=-1)=(1-p(w=0))*0.5
        is_bias = len(shape) == 1
        param_name = 'b' if is_bias else 'W'
        if parameterization == 'shayer':
            param_name_rhoA = '%s_rhoA' % (param_name)
            param_name_rhoB = '%s_rhoB' % (param_name)
        elif parameterization == 'logits':
            param_name_rho = '%s_rho' % (param_name)
        elif parameterization == 'logits_fixedzero':
            param_name_rho_m1 = '%s_rhoM1' % (param_name)
            param_name_rho_p1 = '%s_rhoP1' % (param_name)
        else:
            raise NotImplementedError('Unknown parameterization for ternary weights: \'%s\'' % parameterization)
        param_name_scale_factor = 'scale_factor_rho'
  
        if initial_parameters is None or param_name in initial_parameters:
            if initial_parameters is not None:
                # If both reparameterized parameters and real parameters are
                # provided it is better to raise an exception to make sure that
                # the correct parameters are used for initialization
                if parameterization == 'shayer' and param_name_rhoA in initial_parameters:
                    raise Exception('Initial parameters contain \'%s\' and \'%s\'. Do not know which initial parameters to use'
                                    % (param_name_rhoA, param_name))
                if parameterization == 'shayer' and param_name_rhoB in initial_parameters:
                    raise Exception('Initial parameters contain \'%s\' and \'%s\'. Do not know which initial parameters to use'
                                    % (param_name_rhoB, param_name))
                if parameterization == 'logits' and param_name_rho in initial_parameters:
                    raise Exception('Initial parameters contain \'%s\' and \'%s\'. Do not know which initial parameters to use'
                                    % (param_name_rho, param_name))
                if parameterization == 'logits_fixedzero' and param_name_rho_m1 in initial_parameters:
                    raise Exception('Initial parameters contain \'%s\' and \'%s\'. Do not know which initial parameters to use'
                                    % (param_name_rho_m1, param_name))
                if parameterization == 'logits_fixedzero' and param_name_rho_p1 in initial_parameters:
                    raise Exception('Initial parameters contain \'%s\' and \'%s\'. Do not know which initial parameters to use'
                                    % (param_name_rho_p1, param_name))
                assert param_name_scale_factor not in initial_parameters # scale factors must not be given here
                W_values = LayerLinearForward._assignParameters(initial_parameters, param_name, shape)
                if not (enable_scale_factors and initialization_method_scale_factors == 'kmeans'):
                    # It appears to be better to not normalize the weights if
                    # we are using scale factors initialized with kmeans.
                    #W_values = W_values / np.std(W_values)

                    W_values[W_values <= 0.] = self._mapToEcdf(W_values[W_values <= 0.]) * 1.5 - 1.5 # Use the empirical cdf to initialize
                    W_values[W_values >  0.] = self._mapToEcdf(W_values[W_values >  0.]) * 1.5
            else:
                #print 'Gaussian init'
                W_values = rng.normal(size=shape)
                #print 'CDF like init' (has been temporariliy used in the random-init experiment in the ECML2019 submission)
                #W_values = np.linspace(-1.5, 1.5, np.prod(shape), dtype=theano.config.floatX)
                #rng.shuffle(W_values)
                #W_values = np.reshape(W_values, shape)

            # Compute scale factors. Note that even when no scale factors are
            # used we still compute auxiliary scale factors for initialization
            # of the weight probabilities.
            if enable_scale_factors == True:
                if initialization_method_scale_factors == 'default':
                    w_discrete_values = np.asarray([-1., 0., 1.], dtype=theano.config.floatX)                    
                elif initialization_method_scale_factors == 'kmeans':
                    # TODO: Perform a few random restarts and pick the best result
                    w_discrete_negative, _ = self._kmeansPositiveWeightsFixedZero(-W_values[W_values <= 0], k=1, n_restarts=1, init='random_samples', rng=rng, n_iter=500)
                    w_discrete_positive, _ = self._kmeansPositiveWeightsFixedZero(W_values[W_values > 0], k=1, n_restarts=1, init='random_samples', rng=rng, n_iter=500)
                    w_discrete_values = np.asarray([-w_discrete_negative[1], 0., w_discrete_positive[1]], dtype=theano.config.floatX)
                elif initialization_method_scale_factors == 'kmeans_normalized':
                    # Normalize the weights such that (|w-| + |w+|)/2 == 1 where w- and w+ are obtained using k-means.
                    # Note that the actual procedure works the opposite way by first computing k-means and then normalizing the weights.
                    w_discrete_negative, _ = self._kmeansPositiveWeightsFixedZero(-W_values[W_values <= 0], k=1, n_restarts=1, init='random_samples', rng=rng, n_iter=500)
                    w_discrete_positive, _ = self._kmeansPositiveWeightsFixedZero(W_values[W_values > 0], k=1, n_restarts=1, init='random_samples', rng=rng, n_iter=500)
                    aux = 2. / (w_discrete_negative[1] + w_discrete_positive[1])
                    w_discrete_negative[1] = w_discrete_negative[1] * aux
                    w_discrete_positive[1] = w_discrete_positive[1] * aux
                    W_values = W_values * aux
                    w_discrete_values = np.asarray([-w_discrete_negative[1], 0., w_discrete_positive[1]], dtype=theano.config.floatX)
                else:
                    assert False
                scale_factor_rho_values = np.asarray([-w_discrete_values[0], w_discrete_values[2]], theano.config.floatX)
                scale_factor_rho_values = np.log(np.exp(scale_factor_rho_values) - 1.)
            else:
                w_discrete_values = np.asarray([-1., 0., 1.], dtype=theano.config.floatX)

            if initialization_method in ['shayer', 'expectation']:
                # Method based on matching the expectation of the distribution
                # with given real values.
                # see O. Shayer et al.; Learning Discrete Weights Using the Local Reparameterization Trick; ICLR 2018

                # The following initialization method assumes that w_discrete_values[1]==0.
                p_init_min, p_init_max = 0.05, 0.95

                # Compute p(w = 0)    
                slope1 = (p_init_max - p_init_min) / (-w_discrete_values[0])
                slope2 = (p_init_max - p_init_min) / w_discrete_values[2]
                pA_values = np.zeros(W_values.shape, theano.config.floatX)
                pA_values[W_values <= 0.] = p_init_max + slope1 * W_values[W_values <= 0.]
                pA_values[W_values > 0.] = p_init_max - slope2 * W_values[W_values > 0.]
                pB_values = (W_values / (1. - pA_values) - w_discrete_values[0]) / (-w_discrete_values[0] + w_discrete_values[2])
                pA_values = np.clip(pA_values, p_init_min, p_init_max)
                pB_values = np.clip(pB_values, p_init_min, p_init_max)

                if parameterization in ['logits', 'logits_fixedzero']:
                    p_zr_values = pA_values
                    p_p1_values = (1. - pA_values) * pB_values
                    p_m1_values = (1. - pA_values) * (1. - pB_values)
            elif initialization_method == 'probability':
                # This method initializes the weights to put more probability
                # on values that are close to the given real values without
                # considering the expectation. See quinary weights for the
                # intuition why this method can be useful.
                p_init_min = 0.05
                p_init_max = 1. - p_init_min

                delta_y = p_init_max - p_init_min / 2.
                delta_x = w_discrete_values[1:] - w_discrete_values[:-1]
                slope = delta_y / delta_x
                
                # Compute probabilities p(w = -1)
                idx0 = W_values <= w_discrete_values[0]
                idx1 = W_values > w_discrete_values[1]
                idx2 = np.logical_and(W_values > w_discrete_values[0], W_values <= w_discrete_values[1])
                p_m1_values = np.zeros(W_values.shape)
                p_m1_values[idx0] = p_init_max
                p_m1_values[idx1] = p_init_min / 2.
                p_m1_values[idx2] = p_init_max - slope[0] * (W_values[idx2] - w_discrete_values[0])
                
                # Compute probabilities p(w = 0)
                idx0 = np.logical_or(W_values <= w_discrete_values[0], W_values > w_discrete_values[2])
                idx1 = np.logical_and(W_values > w_discrete_values[0], W_values <= w_discrete_values[1])
                idx2 = np.logical_and(W_values > w_discrete_values[1], W_values <= w_discrete_values[2])
                p_zr_values = np.zeros(W_values.shape)
                p_zr_values[idx0] = p_init_min / 2.
                p_zr_values[idx1] = p_init_min / 2. + slope[0] * (W_values[idx1] - w_discrete_values[0])
                p_zr_values[idx2] = p_init_max - slope[1] * (W_values[idx2] - w_discrete_values[1])
                
                # Compute probabilities p(w = 1)
                idx0 = W_values > w_discrete_values[2]
                idx1 = W_values <= w_discrete_values[1]
                idx2 = np.logical_and(W_values > w_discrete_values[1], W_values <= w_discrete_values[2])
                p_p1_values = np.zeros(W_values.shape)
                p_p1_values[idx0] = p_init_max
                p_p1_values[idx1] = p_init_min / 2.
                p_p1_values[idx2] = p_init_min / 2. + slope[1] * (W_values[idx2] - w_discrete_values[1])

                if parameterization == 'shayer':
                    pA_values = p_zr_values
                    pB_values = (p_p1_values) / (1. - p_zr_values)
            else:
                raise NotImplementedError('Unknown initialization method \'%s\'' % (initialization_method))
            
#             print 'DEBUG (kmeans):', self._kmeansPositiveWeightsFixedZero(W_values[W_values > 0], 2, init='random_samples', rng=rng, n_iter=500)
#             print 'DEBUG (kmeans):', self._kmeansPositiveWeightsFixedZero(-W_values[W_values <= 0], 2, init='random_samples', rng=rng, n_iter=500)

            if parameterization in ['logits', 'logits_fixedzero']:
                debug1 = np.stack([p_m1_values, p_zr_values, p_p1_values], axis=-1)
                debug1 = np.sum(debug1, axis=-1)
                print 'Debug TERNARY weights: Maximum probability discrepancy:', np.max(np.abs(debug1 - 1))

            if parameterization == 'shayer':
                W_rhoA_values = np.asarray(-np.log(1. / pA_values - 1.), dtype=theano.config.floatX)
                W_rhoB_values = np.asarray(-np.log(1. / pB_values - 1.), dtype=theano.config.floatX)
            elif parameterization == 'logits':
                W_rho_values = np.log(np.stack([p_m1_values, p_zr_values, p_p1_values], axis=-1)).astype(theano.config.floatX)
            elif parameterization == 'logits_fixedzero':
                W_rho_values = np.stack([p_m1_values, p_p1_values], axis=-1)
                W_rho_values = (np.log(W_rho_values) - np.log(p_zr_values[..., None])).astype(theano.config.floatX)
                W_rho_m1_values = W_rho_values[..., 0]
                W_rho_p1_values = W_rho_values[..., 1]
            else:
                assert False

        else:
            if parameterization == 'shayer':
                W_rhoA_values = LayerLinearForward._assignParameters(initial_parameters, param_name_rhoA, shape)
                W_rhoB_values = LayerLinearForward._assignParameters(initial_parameters, param_name_rhoB, shape)
            elif parameterization == 'logits':
                W_rho_values = LayerLinearForward._assignParameters(initial_parameters, param_name_rho, shape + (3,))
            elif parameterization == 'logits_fixedzero':
                W_rho_values = LayerLinearForward._assignParameters(initial_parameters, param_name_rho, shape + (2,))
            else:
                assert False
            if enable_scale_factors:
                if param_name_scale_factor in initial_parameters:
                    scale_factor_rho_values = LayerLinearForward._assignParameters(initial_parameters, param_name_scale_factor, (2,))
                else:
                    scale_factor_rho_values = np.full((2,), np.log(np.exp(1.) - 1.), theano.config.floatX)
  
        if parameterization == 'shayer':
            W_rhoA = theano.shared(W_rhoA_values, borrow=True)
            W_rhoB = theano.shared(W_rhoB_values, borrow=True)
            self._addParameterEntry(W_rhoA, param_name_rhoA, is_trainable=True, bounds=self._logit_bounds)
            self._addParameterEntry(W_rhoB, param_name_rhoB, is_trainable=True, bounds=self._logit_bounds)
            W_rhoA_aux = T.nnet.sigmoid(W_rhoA)
            W_rhoA_aux_1m = 1. - W_rhoA_aux
            W_rhoB_aux = T.nnet.sigmoid(W_rhoB)
            W_rhoB_aux_1m = 1. - W_rhoB_aux
            W_p_0 = W_rhoA_aux
            W_p_p1 = W_rhoA_aux_1m * W_rhoB_aux
            W_p_m1 = W_rhoA_aux_1m * W_rhoB_aux_1m
            W_p = T.stack([W_p_m1, W_p_0, W_p_p1], axis=-1) # This is already normalized
        elif parameterization == 'logits':
            W_rho = theano.shared(W_rho_values, borrow=True)
            self._addParameterEntry(W_rho, param_name_rho, is_trainable=True, bounds=self._logit_bounds)
            W_p = W_rho - T.max(W_rho, axis=-1, keepdims=True)
            W_p = W_rho # unsafe, but should be fine with logit_bounds TODO: make a check if logit_bounds are present
            W_p = T.exp(W_p)
            W_p = W_p / T.sum(W_p, axis=-1, keepdims=True)
            W_p_m1 = W_p[..., 0]
            W_p_0 = W_p[..., 1]
            W_p_p1 = W_p[..., 2]
        elif parameterization == 'logits_fixedzero':
            # Similar implementation as for quinary weights (see below)
            W_rho_m1 = theano.shared(W_rho_m1_values, borrow=True)
            W_rho_p1 = theano.shared(W_rho_p1_values, borrow=True)
            self._addParameterEntry(W_rho_m1, param_name_rho_m1, is_trainable=True, bounds=self._logit_bounds)
            self._addParameterEntry(W_rho_p1, param_name_rho_p1, is_trainable=True, bounds=self._logit_bounds)
            W_rho_full = T.stack([W_rho_m1, T.zeros(shape, theano.config.floatX), W_rho_p1], axis=-1)
            W_p = W_rho_full - T.max(W_rho_full, axis=-1, keepdims=True)
            W_p = T.exp(W_p)
            W_p = W_p / T.sum(W_p, axis=-1, keepdims=True)
            W_p_m1 = W_p[..., 0]
            W_p_0 = W_p[..., 1]
            W_p_p1 = W_p[..., 2]
        else:
            assert False
  
        if enable_scale_factors:
            scale_factor_rho = theano.shared(scale_factor_rho_values, 'scale_factor_rho')
            self._addParameterEntry(scale_factor_rho, 'scale_factor_rho', is_trainable=True)
            scale_factor = T.nnet.softplus(scale_factor_rho)
            print 'Debug TERNARY weights: scale_factor.eval()', scale_factor.eval()
            
            W_mean = -W_p_m1 * scale_factor[0] + W_p_p1 * scale_factor[1]
            W_var = W_p_m1 * T.sqr(scale_factor[0] + W_mean) + W_p_0 * T.sqr(W_mean) + W_p_p1 * T.sqr(scale_factor[1] - W_mean)
            # TODO: profile to find a fast implementation of this stuff
            
            ind_pos = T.and_(T.gt(W_p_p1, W_p_m1), T.gt(W_p_p1, W_p_0))
            ind_neg = T.and_(T.le(W_p_p1, W_p_m1), T.gt(W_p_m1, W_p_0))
            W_map = T.zeros(shape, theano.config.floatX)
            W_map = T.switch(ind_pos, scale_factor[1], W_map)
            W_map = T.switch(ind_neg, -scale_factor[0], W_map)
              
            W_sample_values = np.zeros(shape, dtype=theano.config.floatX)
            W_sample = theano.shared(W_sample_values, borrow=True)
            W_sample_cumsum = T.cumsum(W_p, axis=-1)
            W_epsilon = srng.uniform(shape + (1,), dtype=theano.config.floatX)
            W_resampled = T.sum(T.cast(T.le(W_sample_cumsum, W_epsilon), theano.config.floatX), axis=-1) - 1.
            W_resampled = T.switch(T.eq(W_resampled, 1.), W_resampled, scale_factor[1])
            W_resampled = T.switch(T.eq(W_resampled, -1.), W_resampled, -scale_factor[0])
            self._sampling_updates += [(W_sample, W_resampled)]
        else:
            W_mean = -W_p_m1 + W_p_p1
            W_var = W_p_m1 * T.sqr(1. + W_mean) + W_p_0 * T.sqr(W_mean) + W_p_p1 * T.sqr(1. - W_mean)
            W_map = T.cast(T.argmax(W_p, axis=-1), 'float32') - 1.
              
            W_sample_values = np.zeros(shape, dtype=theano.config.floatX)
            W_sample = theano.shared(W_sample_values, borrow=True)
            W_sample_cumsum = T.cumsum(W_p, axis=-1)
            W_epsilon = srng.uniform(shape + (1,), dtype=theano.config.floatX)
            W_resampled = T.sum(T.cast(T.le(W_sample_cumsum, W_epsilon), theano.config.floatX), axis=-1) - 1.
            self._sampling_updates += [(W_sample, W_resampled)]
        print 'Warning: Sampling for distribution \'ternaryShayer\' was not tested'
          
        W_sample_reparam = None
        if self._enable_reparameterization_trick:
            # TODO: Implement Gumbel softmax approximation
            raise NotImplementedError('Reparameterization trick for ternaryShayer not implemented')

        if regularizer == 'shayer':
            assert regularizer_parameters is None
            assert regularizer_weight is not None
            if parameterization == 'shayer':
                cost_regularizer = (T.sum(T.sqr(W_rhoA)) + T.sum(T.sqr(W_rhoB))) * regularizer_weight
            elif parameterization == 'logits':
                cost_regularizer = T.sum(T.sqr(W_rho)) * regularizer_weight
            elif parameterization == 'logits_fixedzero':
                cost_regularizer = (T.sum(T.sqr(W_rho_m1)) + T.sum(T.sqr(W_rho_p1))) * regularizer_weight
            else:
                raise NotImplementedError()
        elif regularizer == 'kl':
            assert regularizer_parameters is not None
            assert regularizer_weight is not None
            cost_entropy = T.sum(W_p * T.log(W_p))
            cost_prior = -(T.log(0.5 * (1. - regularizer_parameters)) * np.prod(shape) + T.log(2. * regularizer_parameters / (1. - regularizer_parameters)) * T.sum(W_p[...,1]))
            cost_regularizer = (cost_entropy + cost_prior) * regularizer_weight
        elif regularizer is None:
            cost_regularizer = 0
        else:
            raise NotImplementedError('Regularizer \'%s\' not implemented' % regularizer)
              
        return W_mean, W_var, W_map, W_sample, W_sample_reparam, cost_regularizer
#################################################################################
#
#################################################################################
    def _addQuaternaryDistribution(self, shape, quaternary_type, parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng):
        if parameterization is None:
            parameterization = 'logits'
        if initialization_method is None:
            initialization_method = 'probability'
            if enable_scale_factors:
                initialization_method_scale_factors = 'default'
        elif enable_scale_factors:
            if isinstance(initialization_method, tuple):
                assert len(initialization_method) == 2
                initialization_method, initialization_method_scale_factors = \
                  initialization_method[0], initialization_method[1]
            else:
                initialization_method_scale_factors = 'default'
        assert quaternary_type in ['symmetric', 'fixed_point_minus', 'fixed_point_plus']
        assert parameterization == 'logits'
        assert enable_scale_factors or initialization_method == 'probability' # [not enable_scale_factors -> init_method == 'probability']

        # prior_param determines the prior probability p(w=0). We have p(w=1)=p(w=-1)=(1-p(w=0))*0.5
        is_bias = len(shape) == 1
        param_name = 'b' if is_bias else 'W'
        if parameterization == 'logits':
            # Optimize logits for w \in {a,b,c,d} # Several options for a,b,c,d are possible
            param_name_rho = '%s_rho' % (param_name)
        else:
            raise NotImplementedError('Unknown parameterization for quinary weights: \'%s\'' % parameterization)
        param_name_scale_factor = 'scale_factor_rho'

        if initial_parameters is None or param_name in initial_parameters:
            if initial_parameters is not None:
                # If both reparameterized parameters and real parameters are
                # provided it is better to raise an exception to make sure that
                # the correct parameters are used for initialization
                if parameterization == 'logits' and param_name_rho in initial_parameters:
                    raise Exception('Initial parameters contain \'%s\' and \'%s\'. Do not know which initial parameters to use'
                                    % (param_name_rho, param_name))
                assert param_name_scale_factor not in initial_parameters # scale factors must not be given here
                W_values = LayerLinearForward._assignParameters(initial_parameters, param_name, shape)
                if not (enable_scale_factors and initialization_method_scale_factors == 'kmeans'):
                    # It appears to be better to not normalize the weights if
                    # we are using scale factors initialized with kmeans.
                    if quaternary_type == 'symmetric':
                        W_values[W_values <= 0.] = self._mapToEcdf(W_values[W_values <= 0.]) * (4./3.) - (4./3.) # Use the empirical cdf to initialize
                        W_values[W_values >  0.] = self._mapToEcdf(W_values[W_values >  0.]) * (4./3.)
                    elif quaternary_type == 'fixed_point_minus':
                        W_values[W_values <= 0.] = self._mapToEcdf(W_values[W_values <= 0.]) * 0.75 - 0.75 # Use the empirical cdf to initialize
                        W_values[W_values >  0.] = self._mapToEcdf(W_values[W_values >  0.]) * 0.75
                    elif quaternary_type == 'fixed_point_plus':
                        W_values[W_values <= 0.] = self._mapToEcdf(W_values[W_values <= 0.]) * 0.75 - 0.75 # Use the empirical cdf to initialize
                        W_values[W_values >  0.] = self._mapToEcdf(W_values[W_values >  0.]) * 0.75
                    else:
                        raise NotImplementedError()
            else:
                W_values = rng.normal(size=shape)

            # Compute scale factors. Note that even when no scale factors are
            # used we still compute auxiliary scale factors for initialization
            # of the weight probabilities.
            if enable_scale_factors == True:
                if initialization_method_scale_factors == 'default':
                    if quaternary_type == 'symmetric':
                        w_discrete_values = np.asarray([-1., -1./3., 1./3., 1.], dtype=theano.config.floatX)
                    elif quaternary_type == 'fixed_point_minus':
                        w_discrete_values = np.asarray([-1., -0.5, 0., 0.5], dtype=theano.config.floatX)
                    elif quaternary_type == 'fixed_point_plus':
                        w_discrete_values = np.asarray([-0.5, 0., 0.5, 1.0], dtype=theano.config.floatX)
                    else:
                        raise NotImplementedError()
                elif initialization_method_scale_factors == 'kmeans':
                    pass
                    # TODO: The following lines are just copied from quinary
                    # TODO: Perform a few random restarts and pick the best result
                    #w_discrete_negative, _ = self._kmeansPositiveWeightsFixedZero(-W_values[W_values <= 0], k=2, n_restarts=1, init='random_samples', rng=rng, n_iter=500)
                    #w_discrete_positive, _ = self._kmeansPositiveWeightsFixedZero(W_values[W_values > 0], k=2, n_restarts=1, init='random_samples', rng=rng, n_iter=500)
                    #w_discrete_values = np.asarray([-w_discrete_negative[2], -w_discrete_negative[1], 0., w_discrete_positive[1], w_discrete_positive[2]], dtype=theano.config.floatX)
                else:
                    assert False
                
                raise NotImplementedError()
                #scale_factor_rho_values = np.asarray([w_discrete_values[1] - w_discrete_values[0],
                #                                      -w_discrete_values[1],
                #                                      w_discrete_values[3],
                #                                      w_discrete_values[4] - w_discrete_values[3]], theano.config.floatX)
                #scale_factor_rho_values = np.log(np.exp(scale_factor_rho_values) - 1.)
                #print 'Debug QUATERNARY weights: w_discrete_values:', w_discrete_values, ', scale_factor_rho_values:', scale_factor_rho_values
            else:
                if quaternary_type == 'symmetric':
                    w_discrete_values = np.asarray([-1., -1./3., 1./3., 1.], dtype=theano.config.floatX)
                elif quaternary_type == 'fixed_point_minus':
                    w_discrete_values = np.asarray([-1., -0.5, 0., 0.5], dtype=theano.config.floatX)
                elif quaternary_type == 'fixed_point_plus':
                    w_discrete_values = np.asarray([-0.5, 0., 0.5, 1.0], dtype=theano.config.floatX)
                else:
                    raise NotImplementedError()
                
            if initialization_method == 'probability':
                # This method initializes the weights to put more probability
                # on values that are close to the given real values without
                # considering the expectation.
                # The 'shayer' method puts too little probability onto the
                # weights {-0.5, +0.5} so that they do not get assigned by the
                # MAP selection procedure.
                p_init_min = 0.05
                p_init_max = 1. - p_init_min

                # NEW IMPLEMENTATION (still testing)
                delta_y = p_init_max - p_init_min / 3.
                delta_x = w_discrete_values[1:] - w_discrete_values[:-1]
                slope = delta_y / delta_x

                # Compute probabilities p(w = w_discrete_values[0])
                idx0 = W_values <= w_discrete_values[0]
                idx1 = W_values > w_discrete_values[1]
                idx2 = np.logical_and(W_values > w_discrete_values[0], W_values <= w_discrete_values[1])
                p_A_values = np.zeros(W_values.shape)
                p_A_values[idx0] = p_init_max
                p_A_values[idx1] = p_init_min / 3.
                p_A_values[idx2] = p_init_max - slope[0] * (W_values[idx2] - w_discrete_values[0])
                 
                # Compute probabilities p(w = w_discrete_values[1])
                idx0 = np.logical_or(W_values <= w_discrete_values[0], W_values > w_discrete_values[2])
                idx1 = np.logical_and(W_values > w_discrete_values[0], W_values <= w_discrete_values[1])
                idx2 = np.logical_and(W_values > w_discrete_values[1], W_values <= w_discrete_values[2])
                p_B_values = np.zeros(W_values.shape)
                p_B_values[idx0] = p_init_min / 3.
                p_B_values[idx1] = p_init_min / 3. + slope[0] * (W_values[idx1] - w_discrete_values[0])
                p_B_values[idx2] = p_init_max - slope[1] * (W_values[idx2] - w_discrete_values[1])
                 
                # Compute probabilities p(w = w_discrete_values[2])
                idx0 = np.logical_or(W_values <= w_discrete_values[1], W_values > w_discrete_values[3])
                idx1 = np.logical_and(W_values > w_discrete_values[1], W_values <= w_discrete_values[2])
                idx2 = np.logical_and(W_values > w_discrete_values[2], W_values <= w_discrete_values[3])
                p_C_values = np.zeros(W_values.shape)
                p_C_values[idx0] = p_init_min / 3.
                p_C_values[idx1] = p_init_min / 3. + slope[1] * (W_values[idx1] - w_discrete_values[1])
                p_C_values[idx2] = p_init_max - slope[2] * (W_values[idx2] - w_discrete_values[2])
                 
                # Compute probabilities p(w = w_discrete_values[3])
                idx0 = W_values > w_discrete_values[3]
                idx1 = W_values <= w_discrete_values[2]
                idx2 = np.logical_and(W_values > w_discrete_values[2], W_values <= w_discrete_values[3])
                p_D_values = np.zeros(W_values.shape)
                p_D_values[idx0] = p_init_max
                p_D_values[idx1] = p_init_min / 3.
                p_D_values[idx2] = p_init_min / 3. + slope[2] * (W_values[idx2] - w_discrete_values[2])
            else:
                raise NotImplementedError('Unknown initialization method \'%s\'' % (initialization_method))

            if parameterization == 'logits':
                print 'Debug init QUATERNARY weights: errs:', np.sum(np.abs(np.sum(np.stack([p_A_values, p_B_values, p_C_values, p_D_values], axis=-1), axis=-1) - 1) >= 1e-6)
                W_rho_values = np.log(np.stack([p_A_values, p_B_values, p_C_values, p_D_values], axis=-1)).astype(theano.config.floatX)
                print 'Debug:', np.max(W_rho_values), np.min(W_rho_values)
            else:
                assert False
        else:
            if parameterization == 'logits':
                W_rho_values = LayerLinearForward._assignParameters(initial_parameters, param_name_rho, shape + (4,))
            else:
                assert False
            if enable_scale_factors:
                if param_name_scale_factor in initial_parameters:
                    scale_factor_rho_values = LayerLinearForward._assignParameters(initial_parameters, param_name_scale_factor, (4,))
                else:
                    raise NotImplementedError()
                    # TODO: Generalize to the different kinds of quaternary weights
                    scale_factor_rho_values = np.full((4,), np.log(np.exp(0.5) - 1.), theano.config.floatX)
  
        if parameterization == 'logits':
            W_rho = theano.shared(W_rho_values, borrow=True)
            self._addParameterEntry(W_rho, param_name_rho, is_trainable=True, bounds=self._logit_bounds)
            W_p = W_rho - T.max(W_rho, axis=-1, keepdims=True)
            W_p = W_rho # unsafe but faster (can be used in conjunction with logit clipping because then there is no risk of overflow)
            W_p = T.exp(W_p)
            W_p = W_p / T.sum(W_p, axis=-1, keepdims=True)
            W_p_A = W_p[..., 0]
            W_p_B = W_p[..., 1]
            W_p_C = W_p[..., 2]
            W_p_D = W_p[..., 3]
        else:
            assert False
        
        # The following lines are for now disabled since they caused crashes on the NVidia V100
        #debug2 = W_p.eval()
        #debug2 = np.sum(debug2, axis=-1)
        # We observed some strange behaviour here. The following line indicated errors although there were none.
        # The errors disappeared when we called W_p.eval() after the line 'W_p = T.exp(W_p)' and before the normalization.
        # This seems to be some kind of weird theano behaviour/bug.
        #print 'Debug QUATERNARY weights: errs:', np.sum(np.abs(debug2 - 1) >= 1e-6), debug2.shape, np.max(np.abs(debug2 - 1))
  
        if enable_scale_factors:
            raise NotImplementedError() # TODO: Implemented for the various kinds of quaternary weights
            scale_factor_rho = theano.shared(scale_factor_rho_values, 'scale_factor_rho')
            self._addParameterEntry(scale_factor_rho, 'scale_factor_rho', is_trainable=True)
            scale_factor = T.nnet.softplus(scale_factor_rho)
            print 'Debug QUATERNARY weights: scale_factor.eval()', scale_factor.eval()
            
            W_mean = -W_p_m1 * (scale_factor[0] + scale_factor[1]) \
                     -W_p_m05 * scale_factor[1] \
                   + W_p_p05 * scale_factor[2] \
                   + W_p_p1 * (scale_factor[2] + scale_factor[3])
            W_var = W_p_m1 * T.sqr(scale_factor[0] + scale_factor[1] + W_mean) \
                  + W_p_m05 * T.sqr(scale_factor[1] + W_mean) \
                  + W_p_0 * T.sqr(W_mean) \
                  + W_p_p05 * T.sqr(scale_factor[2] - W_mean) \
                  + W_p_p1 * T.sqr(scale_factor[2] + scale_factor[3] - W_mean)

            W_map_idx = T.argmax(W_p, axis=-1)
            W_map = T.zeros(shape, theano.config.floatX)
            W_map = T.switch(T.eq(W_map_idx, 0), -scale_factor[0] - scale_factor[1], W_map)
            W_map = T.switch(T.eq(W_map_idx, 1),                  - scale_factor[1], W_map)
            W_map = T.switch(T.eq(W_map_idx, 3),                    scale_factor[2], W_map)
            W_map = T.switch(T.eq(W_map_idx, 4),  scale_factor[3] + scale_factor[2], W_map)

            W_sample_values = np.zeros(shape, dtype=theano.config.floatX)
            W_sample = theano.shared(W_sample_values, borrow=True)
            W_sample_cumsum = T.cumsum(W_p, axis=-1)
            W_epsilon = srng.uniform(shape + (1,), dtype=theano.config.floatX)
            W_resampled_idx = T.sum(T.cast(T.le(W_sample_cumsum, W_epsilon), 'int32'), axis=-1)
            W_resampled = T.zeros(shape, theano.config.floatX)
            W_resampled = T.switch(T.eq(W_resampled_idx, 0), -scale_factor[0] - scale_factor[1], W_map)
            W_resampled = T.switch(T.eq(W_resampled_idx, 1),                  - scale_factor[1], W_map)
            W_resampled = T.switch(T.eq(W_resampled_idx, 3),                    scale_factor[2], W_map)
            W_resampled = T.switch(T.eq(W_resampled_idx, 4),  scale_factor[3] + scale_factor[2], W_map)
            self._sampling_updates += [(W_sample, W_resampled)]
        else:
            if quaternary_type == 'symmetric':
                W_mean = -W_p_A + W_p_D + (1./3.) * (-W_p_B + W_p_C)
                W_var = W_p_A * T.sqr(1. + W_mean) \
                      + W_p_B * T.sqr(1./3. + W_mean) \
                      + W_p_C * T.sqr(1./3. - W_mean) \
                      + W_p_D * T.sqr(1. - W_mean)
                W_map = (T.cast(T.argmax(W_p, axis=-1), 'float32') - 1.5) * (2. / 3.)
            elif quaternary_type == 'fixed_point_minus':
                W_mean = -W_p_A + 0.5 * (-W_p_B + W_p_D)
                W_var = W_p_A * T.sqr(1. + W_mean) \
                      + W_p_B * T.sqr(0.5 + W_mean) \
                      + W_p_C * T.sqr(W_mean) \
                      + W_p_D * T.sqr(0.5 - W_mean)
                W_map = (T.cast(T.argmax(W_p, axis=-1), 'float32') - 2.0) * 0.5
            elif quaternary_type == 'fixed_point_plus':
                W_mean = 0.5 * (W_p_C - W_p_A) + W_p_D
                W_var = W_p_A * T.sqr(0.5 + W_mean) \
                      + W_p_B * T.sqr(W_mean) \
                      + W_p_C * T.sqr(0.5 - W_mean) \
                      + W_p_D * T.sqr(1.0 - W_mean)
                W_map = (T.cast(T.argmax(W_p, axis=-1), 'float32') - 1.0) * 0.5
            else:
                raise NotImplementedError()
              
            W_sample_values = np.zeros(shape, dtype=theano.config.floatX)
            W_sample = theano.shared(W_sample_values, borrow=True)
            W_sample_cumsum = T.cumsum(W_p, axis=-1)
            W_epsilon = srng.uniform(shape + (1,), dtype=theano.config.floatX)
            if quaternary_type == 'symmetric':
                W_resampled = (T.sum(T.cast(T.le(W_sample_cumsum, W_epsilon), theano.config.floatX), axis=-1) - 1.5) * (2. / 3.)
            elif quaternary_type == 'fixed_point_minus':
                W_resampled = (T.sum(T.cast(T.le(W_sample_cumsum, W_epsilon), theano.config.floatX), axis=-1) - 2.0) * 0.5
            elif quaternary_type == 'fixed_point_plus':
                W_resampled = (T.sum(T.cast(T.le(W_sample_cumsum, W_epsilon), theano.config.floatX), axis=-1) - 1.0) * 0.5
            else:
                raise NotImplementedError()
            self._sampling_updates += [(W_sample, W_resampled)]
        print 'Warning: Sampling for distribution \'quaternary\' was not tested'
          
        W_sample_reparam = None
        if self._enable_reparameterization_trick:
            # TODO: Implement Gumbel softmax approximation
            raise NotImplementedError('Reparameterization trick for quinaryShayer not implemented')
          
        if regularizer == 'shayer':
            assert regularizer_parameters is None
            assert regularizer_weight is not None
            if parameterization == 'logits':
                cost_regularizer = T.sum(T.sqr(W_rho)) * regularizer_weight
            else:
                assert False
        elif regularizer == 'kl':
            assert False # TODO: Implement
        elif regularizer == 'kl_gaussprior': # discretized gaussian prior
            assert False # TODO: Implement
        elif regularizer is None:
            cost_regularizer = 0
        else:
            raise NotImplementedError('Regularizer \'%s\' not implemented' % regularizer)
              
        return W_mean, W_var, W_map, W_sample, W_sample_reparam, cost_regularizer
#################################################################################
#
#################################################################################
    def _addQuinaryDistribution(self, shape, parameterization, initialization_method, initial_parameters, regularizer, regularizer_weight, regularizer_parameters, enable_scale_factors, rng, srng):
        if parameterization is None:
            parameterization = 'shayer'
        if initialization_method is None:
            initialization_method = 'probability'
            if enable_scale_factors:
                initialization_method_scale_factors = 'default'
        elif enable_scale_factors:
            if isinstance(initialization_method, tuple):
                assert len(initialization_method) == 2
                initialization_method, initialization_method_scale_factors = \
                  initialization_method[0], initialization_method[1]
            else:
                initialization_method_scale_factors = 'default'

        # prior_param determines the prior probability p(w=0). We have p(w=1)=p(w=-1)=(1-p(w=0))*0.5
        is_bias = len(shape) == 1
        param_name = 'b' if is_bias else 'W'
        if parameterization == 'shayer':
            param_name_rhoA = '%s_rhoA' % (param_name)
            param_name_rhoB = '%s_rhoB' % (param_name)
            param_name_rhoC = '%s_rhoC' % (param_name)
            param_name_rhoD = '%s_rhoD' % (param_name)
        elif parameterization == 'logits':
            # Optimize logits for w \in {-1, -0,5, 0, 0.5, 1}
            param_name_rho = '%s_rho' % (param_name)
        elif parameterization == 'logits_fixedzero':
            # Logits for w \in {-1, -0.5, 0.5, 1}. The logit for w=0 is fixed to zero.
            param_name_rho_m1  = '%s_rhoM1' % (param_name)
            param_name_rho_m05 = '%s_rhoM05' % (param_name)
            param_name_rho_p05 = '%s_rhoP05' % (param_name)
            param_name_rho_p1  = '%s_rhoP1' % (param_name)
        else:
            raise NotImplementedError('Unknown parameterization for quinary weights: \'%s\'' % parameterization)
        param_name_scale_factor = 'scale_factor_rho'

        if initial_parameters is None or param_name in initial_parameters:
            if initial_parameters is not None:
                # If both reparameterized parameters and real parameters are
                # provided it is better to raise an exception to make sure that
                # the correct parameters are used for initialization
                if parameterization == 'shayer' and param_name_rhoA in initial_parameters:
                    raise Exception('Initial parameters contain \'%s\' and \'%s\'. Do not know which initial parameters to use'
                                    % (param_name_rhoA, param_name))
                if parameterization == 'shayer' and param_name_rhoB in initial_parameters:
                    raise Exception('Initial parameters contain \'%s\' and \'%s\'. Do not know which initial parameters to use'
                                    % (param_name_rhoB, param_name))
                if parameterization == 'shayer' and param_name_rhoC in initial_parameters:
                    raise Exception('Initial parameters contain \'%s\' and \'%s\'. Do not know which initial parameters to use'
                                    % (param_name_rhoC, param_name))
                if parameterization == 'shayer' and param_name_rhoD in initial_parameters:
                    raise Exception('Initial parameters contain \'%s\' and \'%s\'. Do not know which initial parameters to use'
                                    % (param_name_rhoD, param_name))
                if parameterization == 'logits' and param_name_rho in initial_parameters:
                    raise Exception('Initial parameters contain \'%s\' and \'%s\'. Do not know which initial parameters to use'
                                    % (param_name_rho, param_name))
                if parameterization == 'logits_fixed_zero' and param_name_rho_m1 in initial_parameters:
                    raise Exception('Initial parameters contain \'%s\' and \'%s\'. Do not know which initial parameters to use'
                                    % (param_name_rho_m1, param_name))
                if parameterization == 'logits_fixed_zero' and param_name_rho_m05 in initial_parameters:
                    raise Exception('Initial parameters contain \'%s\' and \'%s\'. Do not know which initial parameters to use'
                                    % (param_name_rho_m05, param_name))
                if parameterization == 'logits_fixed_zero' and param_name_rho_p05 in initial_parameters:
                    raise Exception('Initial parameters contain \'%s\' and \'%s\'. Do not know which initial parameters to use'
                                    % (param_name_rho_p05, param_name))
                if parameterization == 'logits_fixed_zero' and param_name_rho_p1 in initial_parameters:
                    raise Exception('Initial parameters contain \'%s\' and \'%s\'. Do not know which initial parameters to use'
                                    % (param_name_rho_p1, param_name))
                assert param_name_scale_factor not in initial_parameters # scale factors must not be given here
                W_values = LayerLinearForward._assignParameters(initial_parameters, param_name, shape)
                if not (enable_scale_factors and initialization_method_scale_factors == 'kmeans'):
                    # It appears to be better to not normalize the weights if
                    # we are using scale factors initialized with kmeans.
                    #W_values = W_values / np.std(W_values)
                    
                    #W_values = self._mapToEcdf(W_values) * 2.5 - 1.25 # Use the empirical cdf to initialize
                    
                    W_values[W_values <= 0.] = self._mapToEcdf(W_values[W_values <= 0.]) * 1.25 - 1.25 # Use the empirical cdf to initialize
                    W_values[W_values >  0.] = self._mapToEcdf(W_values[W_values >  0.]) * 1.25
            else:
                W_values = rng.normal(size=shape)

            # Compute scale factors. Note that even when no scale factors are
            # used we still compute auxiliary scale factors for initialization
            # of the weight probabilities.
            if enable_scale_factors == True:
                if initialization_method_scale_factors == 'default':
                    w_discrete_values = np.asarray([-1., -0.5, 0., 0.5, 1.], dtype=theano.config.floatX)                    
                elif initialization_method_scale_factors == 'kmeans':
                    # TODO: Perform a few random restarts and pick the best result
                    w_discrete_negative, _ = self._kmeansPositiveWeightsFixedZero(-W_values[W_values <= 0], k=2, n_restarts=1, init='random_samples', rng=rng, n_iter=500)
                    w_discrete_positive, _ = self._kmeansPositiveWeightsFixedZero(W_values[W_values > 0], k=2, n_restarts=1, init='random_samples', rng=rng, n_iter=500)
                    w_discrete_values = np.asarray([-w_discrete_negative[2], -w_discrete_negative[1], 0., w_discrete_positive[1], w_discrete_positive[2]], dtype=theano.config.floatX)
                else:
                    assert False
                scale_factor_rho_values = np.asarray([w_discrete_values[1] - w_discrete_values[0],
                                                      -w_discrete_values[1],
                                                      w_discrete_values[3],
                                                      w_discrete_values[4] - w_discrete_values[3]], theano.config.floatX)
                scale_factor_rho_values = np.log(np.exp(scale_factor_rho_values) - 1.)
                print 'Debug QUINARY weights: w_discrete_values:', w_discrete_values, ', scale_factor_rho_values:', scale_factor_rho_values
            else:
                w_discrete_values = np.asarray([-1., -0.5, 0., 0.5, 1.], dtype=theano.config.floatX)
                

            if initialization_method in ['shayer', 'expectation']:
                # TODO: Implement for scale factors

                # Method based on matching the expectation of the distribution
                # with given real values.
                # see O. Shayer et al.; Learning Discrete Weights Using the Local Reparameterization Trick; ICLR 2018
                # The following initialization method assumes that w_discrete_values[2]==0.
                p_init_min, p_init_max = 0.05, 0.95

                # Compute p(w = 0)    
                slope1 = (p_init_max - p_init_min) / (-w_discrete_values[0])
                slope2 = (p_init_max - p_init_min) / w_discrete_values[4]
                idx1 = np.logical_and(W_values > w_discrete_values[0], W_values <= 0.)
                idx2 = np.logical_and(W_values > 0., W_values <= w_discrete_values[4])
                pA_values = np.full(W_values.shape, p_init_min, theano.config.floatX)
                pA_values[idx1] = p_init_min + slope1 * (W_values[idx1] - w_discrete_values[0])
                pA_values[idx2] = p_init_max - slope2 * W_values[idx2] # no more clipping needed
                
                # Compute p(w > 0 | w != 0)
                alpha1 = (w_discrete_values[3] + w_discrete_values[4]) * 0.5
                alpha2 = (w_discrete_values[0] + w_discrete_values[1]) * 0.5
                pB_values = (W_values / (1 - pA_values) - alpha2) / (alpha1 - alpha2)
                pB_values = np.clip(pB_values, p_init_min, p_init_max)
                 
                # Compute p(w = w_discrete_values[4] | w > 0)
                pC_values = np.full(pA_values.shape, 0.5)
                pC_values_aux = ((W_values / (1. - pA_values) - 0.5 * (1. - pB_values) * (w_discrete_values[0] + w_discrete_values[1])) / pB_values - w_discrete_values[3]) / (w_discrete_values[4] - w_discrete_values[3])
                pC_values[pB_values == p_init_max] = pC_values_aux[pB_values == p_init_max]
                pC_values = np.clip(pC_values, p_init_min, p_init_max)
                
                # Compute p(w = w_discrete_values[0] | w < 0)
                pD_values = np.full(pA_values.shape, 0.5)
                pD_values_aux = ((W_values / (1. - pA_values) - 0.5 * pB_values * (w_discrete_values[4] + w_discrete_values[3])) / (1. - pB_values) - w_discrete_values[1]) / (w_discrete_values[0] - w_discrete_values[1])
                pD_values[pB_values == p_init_min] = pD_values_aux[pB_values == p_init_min]
                pD_values = np.clip(pD_values, p_init_min, p_init_max)
                
                if parameterization in ['logits', 'logits_fixedzero']:
                    p_zr_values = pA_values
                    p_p1_values = (1. - pA_values) * pB_values * pC_values
                    p_p05_values = (1. - pA_values) * pB_values * (1. - pC_values)
                    p_m1_values = (1. - pA_values) * (1. - pB_values) * pD_values
                    p_m05_values = (1. - pA_values) * (1. - pB_values) * (1. - pD_values)

            elif initialization_method == 'probability':
                # This method initializes the weights to put more probability
                # on values that are close to the given real values without
                # considering the expectation.
                # The 'shayer' method puts too little probability onto the
                # weights {-0.5, +0.5} so that they do not get assigned by the
                # MAP selection procedure.
                p_init_min = 0.05
                p_init_max = 1. - p_init_min

                # NEW IMPLEMENTATION (still testing)
                delta_y = p_init_max - p_init_min / 4.
                delta_x = w_discrete_values[1:] - w_discrete_values[:-1]
                slope = delta_y / delta_x

                # Compute probabilities p(w = -1)
                idx0 = W_values <= w_discrete_values[0]
                idx1 = W_values > w_discrete_values[1]
                idx2 = np.logical_and(W_values > w_discrete_values[0], W_values <= w_discrete_values[1])
                p_m1_values = np.zeros(W_values.shape)
                p_m1_values[idx0] = p_init_max
                p_m1_values[idx1] = p_init_min / 4.
                p_m1_values[idx2] = p_init_max - slope[0] * (W_values[idx2] - w_discrete_values[0])
                 
                # Compute probabilities p(w = -0.5)
                idx0 = np.logical_or(W_values <= w_discrete_values[0], W_values > w_discrete_values[2])
                idx1 = np.logical_and(W_values > w_discrete_values[0], W_values <= w_discrete_values[1])
                idx2 = np.logical_and(W_values > w_discrete_values[1], W_values <= w_discrete_values[2])
                p_m05_values = np.zeros(W_values.shape)
                p_m05_values[idx0] = p_init_min / 4.
                p_m05_values[idx1] = p_init_min / 4. + slope[0] * (W_values[idx1] - w_discrete_values[0])
                p_m05_values[idx2] = p_init_max - slope[1] * (W_values[idx2] - w_discrete_values[1])
                 
                # Compute probabilities p(w = 0)
                idx0 = np.logical_or(W_values <= w_discrete_values[1], W_values > w_discrete_values[3])
                idx1 = np.logical_and(W_values > w_discrete_values[1], W_values <= w_discrete_values[2])
                idx2 = np.logical_and(W_values > w_discrete_values[2], W_values <= w_discrete_values[3])
                p_zr_values = np.zeros(W_values.shape)
                p_zr_values[idx0] = p_init_min / 4.
                p_zr_values[idx1] = p_init_min / 4. + slope[1] * (W_values[idx1] - w_discrete_values[1])
                p_zr_values[idx2] = p_init_max - slope[2] * (W_values[idx2] - w_discrete_values[2])
                 
                # Compute probabilities p(w = 0.5)
                idx0 = np.logical_or(W_values <= w_discrete_values[2], W_values > w_discrete_values[4])
                idx1 = np.logical_and(W_values > w_discrete_values[2], W_values <= w_discrete_values[3])
                idx2 = np.logical_and(W_values > w_discrete_values[3], W_values <= w_discrete_values[4])
                p_p05_values = np.zeros(W_values.shape)
                p_p05_values[idx0] = p_init_min / 4.
                p_p05_values[idx1] = p_init_min / 4. + slope[2] * (W_values[idx1] - w_discrete_values[2])
                p_p05_values[idx2] = p_init_max - slope[3] * (W_values[idx2] - w_discrete_values[3])
                 
                # Compute probabilities p(w = 1)
                idx0 = W_values > w_discrete_values[4]
                idx1 = W_values <= w_discrete_values[3]
                idx2 = np.logical_and(W_values > w_discrete_values[3], W_values <= w_discrete_values[4])
                p_p1_values = np.zeros(W_values.shape)
                p_p1_values[idx0] = p_init_max
                p_p1_values[idx1] = p_init_min / 4.
                p_p1_values[idx2] = p_init_min / 4. + slope[3] * (W_values[idx2] - w_discrete_values[3])

                if parameterization == 'shayer':
                    pA_values = p_zr_values
                    pB_values = (p_p1_values + p_p05_values) / (1. - p_zr_values)
                    pC_values = p_p1_values / (1 - p_zr_values) / pB_values
                    pD_values = p_m1_values / (1 - p_zr_values) / (1. - pB_values)
            else:
                raise NotImplementedError('Unknown initialization method \'%s\'' % (initialization_method))

            if parameterization in ['logits', 'logits_fixedzero']:
                debug1 = np.stack([p_m1_values, p_m05_values, p_zr_values, p_p05_values, p_p1_values], axis=-1)
                debug1 = np.sum(debug1, axis=-1)
                print 'Debug QUINARY weights: Maximum probability discrepancy:', np.max(np.abs(debug1 - 1))

            if parameterization == 'shayer':
                W_rhoA_values = np.asarray(-np.log(1. / pA_values - 1.), dtype=theano.config.floatX)
                W_rhoB_values = np.asarray(-np.log(1. / pB_values - 1.), dtype=theano.config.floatX)
                W_rhoC_values = np.asarray(-np.log(1. / pC_values - 1.), dtype=theano.config.floatX)
                W_rhoD_values = np.asarray(-np.log(1. / pD_values - 1.), dtype=theano.config.floatX)
            elif parameterization == 'logits':
                W_rho_values = np.log(np.stack([p_m1_values, p_m05_values, p_zr_values, p_p05_values, p_p1_values], axis=-1)).astype(theano.config.floatX)
            elif parameterization == 'logits_fixedzero':
                W_rho_values = np.stack([p_m1_values, p_m05_values, p_p05_values, p_p1_values], axis=-1)
                W_rho_values = (np.log(W_rho_values) - np.log(p_zr_values[..., None])).astype(theano.config.floatX)
                W_rho_m1_values = W_rho_values[..., 0]
                W_rho_m05_values = W_rho_values[..., 1]
                W_rho_p05_values = W_rho_values[..., 2]
                W_rho_p1_values = W_rho_values[..., 3]
            else:
                assert False
        else:
            if parameterization == 'shayer':
                W_rhoA_values = LayerLinearForward._assignParameters(initial_parameters, param_name_rhoA, shape)
                W_rhoB_values = LayerLinearForward._assignParameters(initial_parameters, param_name_rhoB, shape)
                W_rhoC_values = LayerLinearForward._assignParameters(initial_parameters, param_name_rhoC, shape)
                W_rhoD_values = LayerLinearForward._assignParameters(initial_parameters, param_name_rhoD, shape)
            elif parameterization == 'logits':
                W_rho_values = LayerLinearForward._assignParameters(initial_parameters, param_name_rho, shape + (5,))
            elif parameterization == 'logits_fixedzero':
                W_rho_values = LayerLinearForward._assignParameters(initial_parameters, param_name_rho, shape + (4,))
            else:
                assert False
            if enable_scale_factors:
                if param_name_scale_factor in initial_parameters:
                    scale_factor_rho_values = LayerLinearForward._assignParameters(initial_parameters, param_name_scale_factor, (4,))
                else:
                    scale_factor_rho_values = np.full((4,), np.log(np.exp(0.5) - 1.), theano.config.floatX)

        if parameterization == 'shayer':
            W_rhoA = theano.shared(W_rhoA_values, borrow=True)
            W_rhoB = theano.shared(W_rhoB_values, borrow=True)
            W_rhoC = theano.shared(W_rhoC_values, borrow=True)
            W_rhoD = theano.shared(W_rhoD_values, borrow=True)
            self._addParameterEntry(W_rhoA, param_name_rhoA, is_trainable=True, bounds=self._logit_bounds)
            self._addParameterEntry(W_rhoB, param_name_rhoB, is_trainable=True, bounds=self._logit_bounds)
            self._addParameterEntry(W_rhoC, param_name_rhoC, is_trainable=True, bounds=self._logit_bounds)
            self._addParameterEntry(W_rhoD, param_name_rhoD, is_trainable=True, bounds=self._logit_bounds)
            W_rhoA_aux = T.nnet.sigmoid(W_rhoA)
            W_rhoA_aux_1m = 1. - W_rhoA_aux
            W_rhoB_aux = T.nnet.sigmoid(W_rhoB)
            W_rhoB_aux_1m = 1. - W_rhoB_aux
            W_rhoC_aux = T.nnet.sigmoid(W_rhoC)
            W_rhoC_aux_1m = 1. - W_rhoC_aux
            W_rhoD_aux = T.nnet.sigmoid(W_rhoD)
            W_rhoD_aux_1m = 1. - W_rhoD_aux
            W_p_0 = W_rhoA_aux
            W_p_p1 = W_rhoA_aux_1m * W_rhoB_aux * W_rhoC_aux
            W_p_p05 = W_rhoA_aux_1m * W_rhoB_aux * W_rhoC_aux_1m
            W_p_m1 = W_rhoA_aux_1m * W_rhoB_aux_1m * W_rhoD_aux
            W_p_m05 = W_rhoA_aux_1m * W_rhoB_aux_1m * W_rhoD_aux_1m
            W_p = T.stack([W_p_m1, W_p_m05, W_p_0, W_p_p05, W_p_p1], axis=-1) # These probabilities are already normalized
        elif parameterization == 'logits':
            W_rho = theano.shared(W_rho_values, borrow=True)
            self._addParameterEntry(W_rho, param_name_rho, is_trainable=True, bounds=self._logit_bounds)
            W_p = W_rho - T.max(W_rho, axis=-1, keepdims=True)
            W_p = W_rho
            W_p = T.exp(W_p)
            W_p = W_p / T.sum(W_p, axis=-1, keepdims=True)
            W_p_m1 = W_p[..., 0]
            W_p_m05 = W_p[..., 1]
            W_p_0 = W_p[..., 2]
            W_p_p05 = W_p[..., 3]
            W_p_p1 = W_p[..., 4]
        elif parameterization == 'logits_fixedzero':
            # The following implementation using four tensors was faster than an
            # implementation where one large tensor combining the four tensors
            # was used.
            W_rho_m1 = theano.shared(W_rho_m1_values, borrow=True)
            W_rho_m05 = theano.shared(W_rho_m05_values, borrow=True)
            W_rho_p05 = theano.shared(W_rho_p05_values, borrow=True)
            W_rho_p1 = theano.shared(W_rho_p1_values, borrow=True)
            self._addParameterEntry(W_rho_m1, param_name_rho_m1, is_trainable=True, bounds=self._logit_bounds)
            self._addParameterEntry(W_rho_m05, param_name_rho_m05, is_trainable=True, bounds=self._logit_bounds)
            self._addParameterEntry(W_rho_p05, param_name_rho_p05, is_trainable=True, bounds=self._logit_bounds)
            self._addParameterEntry(W_rho_p1, param_name_rho_p1, is_trainable=True, bounds=self._logit_bounds)
            W_rho_full = T.stack([W_rho_m1, W_rho_m05, T.zeros(shape, theano.config.floatX), W_rho_p05, W_rho_p1], axis=-1)
            W_p = W_rho_full - T.max(W_rho_full, axis=-1, keepdims=True)
            W_p = T.exp(W_p)
            W_p = W_p / T.sum(W_p, axis=-1, keepdims=True)
            W_p_m1 = W_p[..., 0]
            W_p_m05 = W_p[..., 1]
            W_p_0 = W_p[..., 2]
            W_p_p05 = W_p[..., 3]
            W_p_p1 = W_p[..., 4]
        else:
            assert False

        # The following lines are for now disabled since they caused crashes on the NVidia V100
        #debug2 = W_p.eval()
        #debug2 = np.sum(debug2, axis=-1)
        #print 'Debug QUINARY weights: errs:', np.sum(np.abs(debug2 - 1) >= 1e-6)
  
        if enable_scale_factors:
            scale_factor_rho = theano.shared(scale_factor_rho_values, 'scale_factor_rho')
            self._addParameterEntry(scale_factor_rho, 'scale_factor_rho', is_trainable=True)
            scale_factor = T.nnet.softplus(scale_factor_rho)
            print 'Debug QUINARY weights: scale_factor.eval()', scale_factor.eval()
            
            W_mean = -W_p_m1 * (scale_factor[0] + scale_factor[1]) \
                     -W_p_m05 * scale_factor[1] \
                   + W_p_p05 * scale_factor[2] \
                   + W_p_p1 * (scale_factor[2] + scale_factor[3])
            W_var = W_p_m1 * T.sqr(scale_factor[0] + scale_factor[1] + W_mean) \
                  + W_p_m05 * T.sqr(scale_factor[1] + W_mean) \
                  + W_p_0 * T.sqr(W_mean) \
                  + W_p_p05 * T.sqr(scale_factor[2] - W_mean) \
                  + W_p_p1 * T.sqr(scale_factor[2] + scale_factor[3] - W_mean)

            W_map_idx = T.argmax(W_p, axis=-1)
            W_map = T.zeros(shape, theano.config.floatX)
            W_map = T.switch(T.eq(W_map_idx, 0), -scale_factor[0] - scale_factor[1], W_map)
            W_map = T.switch(T.eq(W_map_idx, 1),                  - scale_factor[1], W_map)
            W_map = T.switch(T.eq(W_map_idx, 3),                    scale_factor[2], W_map)
            W_map = T.switch(T.eq(W_map_idx, 4),  scale_factor[3] + scale_factor[2], W_map)

            W_sample_values = np.zeros(shape, dtype=theano.config.floatX)
            W_sample = theano.shared(W_sample_values, borrow=True)
            W_sample_cumsum = T.cumsum(W_p, axis=-1)
            W_epsilon = srng.uniform(shape + (1,), dtype=theano.config.floatX)
            W_resampled_idx = T.sum(T.cast(T.le(W_sample_cumsum, W_epsilon), 'int32'), axis=-1)
            W_resampled = T.zeros(shape, theano.config.floatX)
            W_resampled = T.switch(T.eq(W_resampled_idx, 0), -scale_factor[0] - scale_factor[1], W_map)
            W_resampled = T.switch(T.eq(W_resampled_idx, 1),                  - scale_factor[1], W_map)
            W_resampled = T.switch(T.eq(W_resampled_idx, 3),                    scale_factor[2], W_map)
            W_resampled = T.switch(T.eq(W_resampled_idx, 4),  scale_factor[3] + scale_factor[2], W_map)
            self._sampling_updates += [(W_sample, W_resampled)]
        else:
            W_mean = -W_p_m1 + W_p_p1 + 0.5 * (-W_p_m05 + W_p_p05)
            W_var = W_p_m1 * T.sqr(1. + W_mean) \
                  + W_p_m05 * T.sqr(0.5 + W_mean) \
                  + W_p_0 * T.sqr(W_mean) \
                  + W_p_p05 * T.sqr(0.5 - W_mean) \
                  + W_p_p1 * T.sqr(1 - W_mean)

            W_map = (T.cast(T.argmax(W_p, axis=-1), 'float32') - 2.) * 0.5
              
            W_sample_values = np.zeros(shape, dtype=theano.config.floatX)
            W_sample = theano.shared(W_sample_values, borrow=True)
            W_sample_cumsum = T.cumsum(W_p, axis=-1)
            W_epsilon = srng.uniform(shape + (1,), dtype=theano.config.floatX)
            W_resampled = (T.sum(T.cast(T.le(W_sample_cumsum, W_epsilon), theano.config.floatX), axis=-1) - 2.) * 0.5
            self._sampling_updates += [(W_sample, W_resampled)]
        print 'Warning: Sampling for distribution \'quinary\' was not tested'
          
        W_sample_reparam = None
        if self._enable_reparameterization_trick:
            # TODO: Implement Gumbel softmax approximation
            raise NotImplementedError('Reparameterization trick for quinaryShayer not implemented')
          
        if regularizer == 'shayer':
            assert regularizer_parameters is None
            assert regularizer_weight is not None
            if parameterization == 'shayer':
                cost_regularizer = (T.sum(T.sqr(W_rhoA)) + \
                                    T.sum(T.sqr(W_rhoB)) + \
                                    T.sum(T.sqr(W_rhoC)) + \
                                    T.sum(T.sqr(W_rhoD))) * regularizer_weight
            elif parameterization == 'logits':
                cost_regularizer = T.sum(T.sqr(W_rho)) * regularizer_weight
            elif parameterization == 'logits_fixedzero':
                cost_regularizer = (T.sum(T.sqr(W_rho_m1)) + \
                                    T.sum(T.sqr(W_rho_m05)) + \
                                    T.sum(T.sqr(W_rho_p05)) + \
                                    T.sum(T.sqr(W_rho_p1))) * regularizer_weight
            else:
                assert False
        elif regularizer == 'kl':
            assert regularizer_parameters is not None
            assert regularizer_weight is not None
            assert len(regularizer_parameters) in [3,5]
            if len(regularizer_parameters) == 3:
                prior_p = np.asarray([regularizer_parameters[2],
                                      regularizer_parameters[1],
                                      regularizer_parameters[0],
                                      regularizer_parameters[1],
                                      regularizer_parameters[2]], 'float64')
            elif len(regularizer_parameters) == 5:
                prior_p = np.asarray([regularizer_parameters], 'float64')
            assert np.abs(np.sum(prior_p) - 1) <= 1e-6
            log_prior_p = np.log(prior_p).astype(theano.config.floatX)
            cost_regularizer = T.sum(W_p * (T.log(W_p) - log_prior_p)) * regularizer_weight
        elif regularizer == 'kl_gaussprior': # discretized gaussian prior
            assert regularizer_parameters is not None
            assert regularizer_weight is not None
            log_prior_p = -0.5 * np.asarray([-1, -0.5, 0., 0.5, 1.]) / regularizer_parameters
            log_prior_p = log_prior_p - np.max(log_prior_p)
            log_prior_p = log_prior_p - np.log(np.sum(np.exp(log_prior_p)))
            log_prior_p = log_prior_p.astype(theano.config.floatX)
            cost_regularizer = T.sum(W_p * (T.log(W_p) - log_prior_p)) * regularizer_weight
        elif regularizer is None:
            cost_regularizer = 0
        else:
            raise NotImplementedError('Regularizer \'%s\' not implemented' % regularizer)
              
        return W_mean, W_var, W_map, W_sample, W_sample_reparam, cost_regularizer
        
    @staticmethod
    def _assignParameters(params, param_name, shape):
        if param_name not in params:
                raise Exception('Initialization parameter \'%s\' not found' % param_name)
        if params[param_name].shape != shape:
            raise Exception('Initialization parameter \'%s\' must have shape (%s) but has shape (%s)' 
                % (param_name, ','.join(map(str,shape)), ','.join(map(str, params[param_name].shape))))
        return params[param_name].astype(theano.config.floatX)

    @staticmethod
    def _kmeansPositiveWeightsFixedZero(weights, k, n_restarts=1, init='linear', rng=None, n_iter=100):
        '''
        Calls _kmeansPositiveWeightsFixedZeroImpl n_restarts times
        '''
        best_cumdist = np.Inf
        best_centers = None
        for _ in range(n_restarts):
            centers, cumdist = LayerLinearForward._kmeansPositiveWeightsFixedZeroImpl(weights, k, init='linear', rng=None, n_iter=100)
            if cumdist < best_cumdist:
                best_cumdist = cumdist
                best_centers = centers
        return best_centers, best_cumdist
    
    @staticmethod
    def _kmeansPositiveWeightsFixedZeroImpl(weights, k, init='linear', rng=None, n_iter=100):
        '''
        k-means algorithm for positive weights. This k-means algorithm assigns
        a fixed cluster at w=0 and finds k additional cluster centers.
        '''
        assert np.all(weights >= 0.)
        assert init in ['linear', 'random_samples']
        assert init != 'random_samples' or rng is not None
        
        if k > 2:
            print 'Warning: kmeans with k>2: Too large values for k could produce NaNs'
        
        weights = np.reshape(weights, (-1, 1))
        if init == 'linear':
            centers = np.linspace(0., np.max(weights), k + 1).astype(weights.dtype)
        elif init == 'random_samples':
            centers = rng.choice(weights.flatten(), size=k, replace=False)
            centers = np.sort(centers)
            centers = np.concatenate([np.asarray([0.], centers.dtype), centers])
        else:
            assert False
     
        center_idx_old = np.full(weights.size, -1, 'int32')
        for _ in range(n_iter):
            # Find next cluster centers
            center_idx = np.argmin(np.abs(weights - centers), axis=1) # Note: weights is Nx1 --> broadcasting
            if np.all(center_idx_old == center_idx):
                break
            center_idx_old = center_idx

            # Compute means
            for i in range(1, k+1):
                centers[i] = np.mean(weights[center_idx == i])
        
        cumdist = 0.
        for i in range(k+1):
            cumdist += np.sum((weights[center_idx == i] - centers[i]) ** 2.)
                
        return centers, cumdist

    @staticmethod
    def _mapToEcdf(w):
        '''
        Assigns each weight to the value of the empirical cdf \in [0,1]. The weights are basically sorted
        and stored with equal distances to their neighbors in this sorted list.
        '''
        xs, ys = np.unique(w, return_counts=True) # np.unique implicitly sorts and flattens w
        ys = np.asarray(ys, dtype=np.float64) # float64 is important here
        ys = np.cumsum(ys)
        ys /= ys[-1]
        f = interp1d(xs, ys, kind='linear')
        return f(w)
