'''
Abstract layer class that is implemented by all network layers
'''

from abc import ABCMeta, abstractmethod

class Layer:
    __metaclass__ = ABCMeta
    
#     VALUE_TYPE_BINARY = 'binary'
#     VALUE_TYPE_TERNARY = 'ternary'
#     VALUE_TYPE_DISCRETE = 'discrete'
#     VALUE_TYPE_REAL = 'real'

    def __init__(self, input_layer, layer_type):
        self._input_layer = input_layer
        self._layer_type = layer_type
        self._parameter_entries = []
    
    #---------------------------------------------------------------------------
    # methods regarding the output
    @abstractmethod
    def getTrainOutput(self):
        raise NotImplementedError('must be implemented by subclass')
    
    @abstractmethod
    def getPredictionOutput(self):
        raise NotImplementedError('must be implemented by subclass')
    
    @abstractmethod
    def getSampleOutput(self):
        raise NotImplementedError('must be implemented by subclass')
    #---------------------------------------------------------------------------
    
    def getTrainUpdates(self):
        '''
        Returns Theano-updates that can be added to training updates. So far it
        is just intended for batch normalization.
        '''
        return self._input_layer.getTrainUpdates()
    
    def isOutputDistribution(self):
        '''
        Returns true if the output of this layer is a distribution, and false if
        the output of this layer are deterministic values.
        '''
        return self._input_layer.isOutputDistribution()
    
    def isOutputFeatureMap(self):
        '''
        Returns true if the output of this layer are feature maps, i.e., image
        data with spatial organization, and false if the output are plain
        features.
        '''
        return self._input_layer.isOutputFeatureMap()
    
    def getOutputType(self):
        '''
        Returns the type of which is one of
         - 'real': Could be any real value
         - 'binary': Either -1 or +1
        Must be implemented by subclass
        '''
        raise NotImplementedError('getOutputType must be implemented by subclass')
    
    def getCost(self):
        return self._input_layer.getCost()
    
    def getOutputShape(self):
        '''
        Returns the shape of the output as a tuple (without the number of
        samples in the mini-batches).
        '''
        return self._input_layer.getOutputShape()
    
    def getParameters(self):
        parameters = [p['param'] for p in self._parameter_entries]
        return self._input_layer.getParameters() + parameters
    
    def getParameterEntries(self):
        return self._input_layer.getParameterEntries() + self._parameter_entries
    
    def getLayerParameters(self):
        parameters = [p['param'] for p in self._parameter_entries]
        return parameters
    
    def getLayerParameterEntries(self):
        return self._parameter_entries
    
    def getSymbolicWeights(self, weight_types):
        return self._input_layer.getSymbolicWeights(weight_types)
    
    def getTrainOutputsNames(self):
        return self._input_layer.getTrainOutputsNames() + [(self._layer_type, self.getTrainOutput())]
    
    def getPredictionOutputsNames(self):
        return self._input_layer.getPredictionOutputsNames() + [(self._layer_type, self.getPredictionOutput())]
    
    def getSampleOutputsNames(self):
        return self._input_layer.getSampleOutputsNames() + [(self._layer_type, self.getSampleOutput())]
    
    def getSymbolicInput(self):
        return self._input_layer.getSymbolicInput()
    
    def getSymbolicTarget(self):
        return None
    
    def getMessage(self):
        '''
        Prints a message of this layer with useful information
        '''
        pass

    def getLayerSpecificValues(self, layer_type):
        return self._input_layer.getLayerSpecificValues(layer_type)

    def _addParameterEntry(self, param, param_name, is_trainable=True, step_size_scaler=1., bounds=None):
        param_entry = {'layer'            : self,
                       'param'            : param,
                       'name'             : param_name,
                       'trainable'        : is_trainable,
                       'step_size_scaler' : step_size_scaler,
                       'bounds'           : bounds}
        self._parameter_entries.append( param_entry )
