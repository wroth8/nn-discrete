import numpy as np

from datasets.mnist import downloadMnist
from os.path import isfile

from data.DefaultDataLoader import DefaultDataLoader
from network.LayerInput import LayerInput
from network.LayerOutput import LayerOutput
from network.LayerDropout import LayerDropout
from network.LayerFC import LayerFC
from network.LayerBatchnorm import LayerBatchnorm
from network.LayerActivationTanh import LayerActivationTanh
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


def getMnistPIModel(rng, srng):
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

    # Layer 1: FC1200
    layer = LayerInput((784,))
    layer = LayerDropout(layer, 0.2, srng)
    layer = LayerFC(layer, 1200, rng, srng, weight_type='real', regularizer='l2', regularizer_weight=weight_decay,
                    initial_parameters=None)
    p_dict = addParametersToDict(layer, 0, p_dict)
    layer = LayerBatchnorm(layer, alpha=0.1, initial_parameters=None)
    p_dict = addParametersToDict(layer, 0, p_dict)
    layer = LayerActivationTanh(layer)

    # Layer 2: FC1200
    layer = LayerDropout(layer, 0.4, srng)
    layer = LayerFC(layer, 1200, rng, srng, weight_type='real', regularizer='l2', regularizer_weight=weight_decay,
                    initial_parameters=None)
    p_dict = addParametersToDict(layer, 1, p_dict)
    layer = LayerBatchnorm(layer, alpha=0.1, initial_parameters=None)
    p_dict = addParametersToDict(layer, 1, p_dict)
    layer = LayerActivationTanh(layer)

    # Layer 3: FC10
    layer = LayerDropout(layer, 0.4, srng)
    layer = LayerFC(layer, 10, rng, srng, weight_type='real', regularizer='l2',
                    regularizer_weight=weight_decay, enable_bias=True, bias_type='real', initial_parameters=None)
    p_dict = addParametersToDict(layer, 2, p_dict)
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
    dataset_file = 'mnist.npz'
    if not isfile(dataset_file):
        downloadMnist(dataset_file)
    
    data = dict(np.load(dataset_file))
    data['x_tr'] = data['x_tr'] * 2.0 - 1.0
    data['x_va'] = data['x_va'] * 2.0 - 1.0
    data['x_te'] = data['x_te'] * 2.0 - 1.0
    data['t_tr'] = data['t_tr'].astype(np.int32)
    data['t_va'] = data['t_va'].astype(np.int32)
    data['t_te'] = data['t_te'].astype(np.int32)

    rng = np.random.RandomState()
    srng = RandomStreamsGPU(rng.randint(1, 2147462579, size=(6,)))

    # Setup data loaders
    train_generator = DefaultDataLoader(data['x_tr'], data['t_tr'], 100, rng=rng)
    validation_generator = DefaultDataLoader(data['x_va'], data['t_va'], 100)
    test_generator = DefaultDataLoader(data['x_te'], data['t_te'], 100)

    # Create model
    global parameters
    layer, parameters = getMnistPIModel(rng, srng)

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
                            'cooldown' : 150,
                            'patience' : 25,
                            'factor'   : 0.5},
        n_epochs=1000,
        callback_validation_error_decreased=[(cbErrVaDecreased, [])])

    # Store model parameters. The model parameters of the best model according to the validation error are now in
    # p_vals.
    model_file = 'mnist_pi_model_real.npz'
    print 'Optimization finished. Storing model parameters to ''%s''' % model_file
    np.savez_compressed(model_file, **p_vals)


if __name__ == '__main__':
    main()
