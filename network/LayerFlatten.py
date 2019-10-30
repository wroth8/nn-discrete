'''
Flattens a convolutional layer for the transition to a fully-connected layer.
'''

from Layer import Layer
import theano.tensor as T

class LayerFlatten(Layer):
    def __init__(self,
                 input_layer):
        super(LayerFlatten, self).__init__(input_layer, 'flatten')
        # This layer only makes sense when it is used on feature map
        assert input_layer.isOutputFeatureMap()

        shape_in = input_layer.getOutputShape()
        self._n_neurons = shape_in[0] * shape_in[1] * shape_in[2]

    def getTrainOutput(self):
        if self._input_layer.isOutputDistribution():
            x_in_mean, x_in_var = self._input_layer.getTrainOutput()
            x_in_mean = T.reshape(x_in_mean, (x_in_mean.shape[0], self._n_neurons))
            x_in_var = T.reshape(x_in_var, (x_in_var.shape[0], self._n_neurons))
            return x_in_mean, x_in_var
        else:
            x_in = self._input_layer.getTrainOutput()
            x_in = T.reshape(x_in, (x_in.shape[0], self._n_neurons))
            return x_in

    def getPredictionOutput(self):
        x_in = self._input_layer.getPredictionOutput()
        x_in = T.reshape(x_in, (x_in.shape[0], self._n_neurons))
        return x_in

    def getSampleOutput(self):
        x_in = self._input_layer.getSampleOutput()
        x_in = T.reshape(x_in, (x_in.shape[0], self._n_neurons))
        return x_in
    
    def isOutputFeatureMap(self):
        return False
    
    def getOutputType(self):
        return self._input_layer.getOutputType()
    
    def getOutputShape(self):
        return (self._n_neurons,)
    
    def getMessage(self):
        return self._input_layer.getMessage() + '\n%20s: OutputShape=%15s, DistributionOutput=%d' % ('LayerFlatten', str(self.getOutputShape()), self.isOutputDistribution())

