'''
The input layer.
'''

from abc import abstractmethod
from Layer import Layer

import theano
import theano.tensor as T

class LayerInput(Layer):
    def __init__(self,
                 shape,
                 value_type=None):
        super(LayerInput, self).__init__(None, 'input')

        assert len(shape) == 1 or len(shape) == 3
        # if len(shape) == 1
        #   shape[0]: number of features
        # if len(shape) == 3:
        #   shape[0]: number of feature maps
        #   shape[1]: rows
        #   shape[2]: cols
        self._shape = shape
        self._value_type = value_type
        if len(shape) == 1:
            self.x = T.fmatrix('x')
        else:
            self.x = T.tensor4('x')

    def getTrainOutput(self):
        return self.x
    
    def getPredictionOutput(self):
        return self.x
    
    def getSampleOutput(self):
        return self.x
    
    def getTrainUpdates(self):
        return []
        
    def isOutputDistribution(self):
        return False
    
    def isOutputFeatureMap(self):
        return len(self._shape) == 3
    
    def getOutputType(self):
        return 'real'

    def getCost(self):
        return T.constant(0, dtype=theano.config.floatX)

    def getOutputShape(self):
        return self._shape
    
    def getParameters(self):
        return []
    
    def getParameterEntries(self):
        return []
    
    def getSymbolicWeights(self, weight_types):
        return []
    
    def getSymbolicInput(self):
        return self.x
    
    def getTrainOutputsNames(self):
        return [(self._layer_type, self.x)]
    
    def getPredictionOutputsNames(self):
        return [(self._layer_type, self.x)]
    
    def getSampleOutputsNames(self):
        return [(self._layer_type, self.x)]

    def getLayerSpecificValues(self, layer_type):
        return {}
    
    def getMessage(self):
        return '%20s: OutputShape=%15s' % ('LayerInput', str(self._shape))
