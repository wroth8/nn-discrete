import numpy as np

from datasets.svhn import downloadSvhn
from os.path import isfile

from data.Uint8PixelDataLoader import Uint8PixelDataLoader
from network.LayerInput import LayerInput
from network.LayerOutput import LayerOutput
from network.LayerDropout import LayerDropout
from network.LayerFC import LayerFC
from network.LayerBatchnorm import LayerBatchnorm
from network.LayerActivationTanh import LayerActivationTanh
from network.LayerConv import LayerConv
from network.LayerPooling import LayerPooling
from network.LayerFlatten import LayerFlatten
from optimization.optimizeNetwork import optimizeNetwork

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreamsGPU


def addParametersToDict(layer, layer_idx, p_dict, prefix=''):
    '''
    Adds the layer parameters to a dict with the key being the parameter name and the layer index. This is just to
    distinguish Layer objects because they do not contain the layer index.
    '''
    for p in layer.getLayerParameterEntries():
        p_dict['%s%s_%d' % (prefix, p['name'], layer_idx)] = p['param']
    return p_dict


def getSvhnModel(rng, srng):
    '''
    Constructs a real-valued CNN model.

    @param rng: Numpy rng object
    @param srng: rng object used by theano
    @return layer: The CNN model
    @return p_dict: Dictionary containing the shared variables with the model parameters. This helps to easily store
        parameters to a file during/after training
    '''
    weight_decay = 1e-4
    p_dict = {}

    # Layer 1: 64C3
    layer = LayerInput((3,32,32))
    layer = LayerConv(layer, 64, (3,3), (1,1), 'half', rng, srng, weight_type='real', regularizer='l2',
                      regularizer_weight=weight_decay, initial_parameters=None)
    p_dict = addParametersToDict(layer, 0, p_dict)
    layer = LayerBatchnorm(layer, alpha=0.1, initial_parameters=None)
    p_dict = addParametersToDict(layer, 0, p_dict)
    layer = LayerActivationTanh(layer)

    # Layer 2: 64C3-P2
    layer = LayerDropout(layer, 0.2, srng)
    layer = LayerConv(layer, 64, (3,3), (1,1), 'half', rng, srng, weight_type='real', regularizer='l2',
                      regularizer_weight=weight_decay, initial_parameters=None)
    p_dict = addParametersToDict(layer, 1, p_dict)
    layer = LayerBatchnorm(layer, alpha=0.1, initial_parameters=None)
    p_dict = addParametersToDict(layer, 1, p_dict)
    layer = LayerActivationTanh(layer)
    layer = LayerPooling(layer, (2,2), mode='max')

    # Layer 3: 128C3
    layer = LayerDropout(layer, 0.2, srng)
    layer = LayerConv(layer, 128, (3,3), (1,1), 'half', rng, srng, weight_type='real', regularizer='l2',
                      regularizer_weight=weight_decay, initial_parameters=None)
    p_dict = addParametersToDict(layer, 2, p_dict)
    layer = LayerBatchnorm(layer, alpha=0.1, initial_parameters=None)
    p_dict = addParametersToDict(layer, 2, p_dict)
    layer = LayerActivationTanh(layer)

    # Layer 4: 128C3-P2
    layer = LayerDropout(layer, 0.3, srng)
    layer = LayerConv(layer, 128, (3,3), (1,1), 'half', rng, srng, weight_type='real', regularizer='l2',
                      regularizer_weight=weight_decay, initial_parameters=None)
    p_dict = addParametersToDict(layer, 3, p_dict)
    layer = LayerBatchnorm(layer, alpha=0.1, initial_parameters=None)
    p_dict = addParametersToDict(layer, 3, p_dict)
    layer = LayerActivationTanh(layer)
    layer = LayerPooling(layer, (2,2), mode='max')
    
    # Layer 5: 256C3
    layer = LayerDropout(layer, 0.3, srng)
    layer = LayerConv(layer, 256, (3,3), (1,1), 'half', rng, srng, weight_type='real', regularizer='l2',
                      regularizer_weight=weight_decay, initial_parameters=None)
    p_dict = addParametersToDict(layer, 4, p_dict)
    layer = LayerBatchnorm(layer, alpha=0.1, initial_parameters=None)
    p_dict = addParametersToDict(layer, 4, p_dict)
    layer = LayerActivationTanh(layer)

    # Layer 6: 256C3-P2
    layer = LayerDropout(layer, 0.3, srng)
    layer = LayerConv(layer, 256, (3,3), (1,1), 'half', rng, srng, weight_type='real', regularizer='l2',
                      regularizer_weight=weight_decay, initial_parameters=None)
    p_dict = addParametersToDict(layer, 5, p_dict)
    layer = LayerBatchnorm(layer, alpha=0.1, initial_parameters=None)
    p_dict = addParametersToDict(layer, 5, p_dict)
    layer = LayerActivationTanh(layer)
    layer = LayerPooling(layer, (2,2), mode='max')
    layer = LayerFlatten(layer)

    # Layer 7: FC1024
    layer = LayerDropout(layer, 0.4, srng)
    layer = LayerFC(layer, 1024, rng, srng, weight_type='real', regularizer='l2',
                    regularizer_weight=weight_decay, initial_parameters=None)
    p_dict = addParametersToDict(layer, 6, p_dict)
    layer = LayerBatchnorm(layer, alpha=0.1, initial_parameters=None)
    p_dict = addParametersToDict(layer, 6, p_dict)
    layer = LayerActivationTanh(layer)

    # Layer 8: FC10
    layer = LayerFC(layer, 10, rng, srng, weight_type='real', regularizer='l2',
                    regularizer_weight=weight_decay, enable_bias=True, bias_type='real', initial_parameters=None)
    p_dict = addParametersToDict(layer, 7, p_dict)
    layer = LayerOutput(layer, 10, objective='crossentropy')
    
    return layer, p_dict


def cbValidationErrorDecreased():
    '''
    Callback function that is called when the error on the validation set has decreased. Stores the model parameters.
    '''
    global p_vals, parameters
    p_vals = {}
    for p in parameters:
        p_vals[p] = parameters[p].get_value()
    print 'Validation error decreased --> storing the model'


def main():
    dataset_file = 'svhn.npz'
    if not isfile(dataset_file):
        downloadSvhn(dataset_file)
    
    data = dict(np.load(dataset_file))
    data['t_tr'] = data['t_tr'].astype(np.int32)
    data['t_va'] = data['t_va'].astype(np.int32)
    data['t_te'] = data['t_te'].astype(np.int32)

    rng = np.random.RandomState()
    srng = RandomStreamsGPU(rng.randint(1, 2147462579, size=(6,)))

    # Setup data loaders
    train_generator = Uint8PixelDataLoader(data['x_tr'], data['t_tr'], 250, normalization='m1_p1')
    validation_generator = Uint8PixelDataLoader(data['x_va'], data['t_va'], 250, normalization='m1_p1')
    test_generator = Uint8PixelDataLoader(data['x_te'], data['t_te'], 250, normalization='m1_p1')

    # Create model
    global parameters
    layer, parameters = getSvhnModel(rng, srng)

    # Do optimization
    print layer.getMessage()
    cbErrVaDecreased = lambda : cbValidationErrorDecreased()

    global p_vals
    optimizeNetwork(layer,
        loader_tr=train_generator,
        loader_va=validation_generator,
        loader_te=test_generator,
        optimization_algorithm='adam',
        step_size=1e-3,
        step_size_scale_fn={'type'     : 'plateau',
                            'monitor'  : 'ce_va',
                            'cooldown' : 30,
                            'patience' : 10,
                            'factor'   : 0.5},
        n_epochs=300,
        callback_validation_error_decreased=[(cbErrVaDecreased, [])])

    # Store model parameters. The model parameters of the best model according to the validation error are now in
    # p_vals.
    model_file = 'svhn_model_real.npz'
    print 'Optimization finished. Storing model parameters to ''%s''' % model_file
    np.savez_compressed(model_file, **p_vals)


if __name__ == '__main__':
    main()
