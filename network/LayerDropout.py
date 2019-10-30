'''
Performs dropout with Bernoulli noise.
'''

from abc import abstractmethod
from Layer import Layer

import theano
import theano.tensor as T
import numpy as np
import math

class LayerDropout(Layer):
    def __init__(self,
                 input_layer,
                 p_dropout,
                 srng,
                 enable_prediction_sampling=False,
                 enable_mean_correction=True,
                 dropout_per_sample=True, # if True, each sample will be individually dropped out
                 dropout_mode='full'):
        '''
        enable_prediction_sampling: Determines whether dropout sampling should
            be applied for predictions.
        enable_mean_correction: Determines whether the output for dropout for
            predictions should be normalized to satisfy the mean activation.
        '''
        super(LayerDropout, self).__init__(input_layer, 'dropout')
        # If prediction sampling is enabled, there is no need for mean correction
        assert not enable_prediction_sampling or not enable_mean_correction
        # If the input is not a feature map, dropout_mode can only be full
        assert self._input_layer.isOutputFeatureMap() or dropout_mode == 'full'
        # If the input is a feature map, dropout_mode must be in ['full', 'feature', 'pixel']
        assert not self._input_layer.isOutputFeatureMap() or dropout_mode in ['full', 'featuremap', 'pixel']

        self._p_dropout = p_dropout
        self._srng = srng
        self._enable_prediction_sampling = enable_prediction_sampling
        self._enable_mean_correction = enable_mean_correction
        self._dropout_per_sample = dropout_per_sample
        self._dropout_mode = dropout_mode

    def getTrainOutput(self):
        if self._input_layer.isOutputDistribution():
            x_in_mean, x_in_var = self._input_layer.getTrainOutput()
            dropout_mask = self._getDropoutMask(x_in_mean)
            x_out_mean = x_in_mean * dropout_mask
            x_out_var = x_in_var * dropout_mask
            return x_out_mean, x_out_var
        else:
            x_in = self._input_layer.getTrainOutput()
            dropout_mask = self._getDropoutMask(x_in)
            x_out = x_in * dropout_mask
            return x_out
    
    def getPredictionOutput(self):
        x_in = self._input_layer.getPredictionOutput()
        if self._enable_prediction_sampling:
            dropout_mask = self._getDropoutMask(x_in)
            x_out = x_in * dropout_mask
        elif self._enable_mean_correction:
            x_out = x_in * (1. - self._p_dropout)
        else:
            x_out = x_in
        return x_out
    
    def getSampleOutput(self):
        x_in = self._input_layer.getSampleOutput()
        if self._enable_prediction_sampling:
            dropout_mask = self._getDropoutMask(x_in)
            x_out = x_in * dropout_mask
        elif self._enable_mean_correction:
            x_out = x_in * (1. - self._p_dropout)
        else:
            x_out = x_in
        return x_out
    
    def getOutputType(self):
        return 'real'
    
    def getMessage(self):
        return self._input_layer.getMessage() + '\n%20s: OutputShape=%15s, DistributionOutput=%d, pDropout=%f' % ('LayerDropout', str(self.getOutputShape()), self.isOutputDistribution(), self._p_dropout)
    
    def _getDropoutMask(self, x_in):
        if self.isOutputFeatureMap():
            assert self._dropout_mode in ['full', 'featuremap', 'pixel']
            dropout_mask_shape0 = x_in.shape[0] if self._dropout_per_sample else 1 # If dropout_per_sample==1 then use a different mask for each sample
            if self._dropout_mode == 'full' and self._dropout_per_sample:
                dropout_mask_shape = x_in.shape # Avoid an unnecessary T.stack in the computation graph for this most common case
            elif self._dropout_mode == 'full':
                dropout_mask_shape = T.stack([dropout_mask_shape0, x_in.shape[1], x_in.shape[2], x_in.shape[3]], axis=0)
            elif self._dropout_mode == 'featuremap':
                dropout_mask_shape = T.stack([dropout_mask_shape0, x_in.shape[1], 1, 1], axis=0)
            elif self._dropout_mode == 'pixel':
                dropout_mask_shape = T.stack([dropout_mask_shape0, 1, x_in.shape[2], x_in.shape[3]], axis=0)
        else:
            assert self._dropout_mode == 'full'
            if self._dropout_per_sample:
                dropout_mask_shape = x_in.shape
            else:
                dropout_mask_shape = T.stack([1, x_in.shape[1]])

        return T.switch(self._srng.uniform(dropout_mask_shape) < self._p_dropout, 0., 1.)
