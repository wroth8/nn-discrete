import numpy as np

from datasets.cifar100 import downloadCifar100
from os.path import isfile

from data.DefaultDataLoader import DefaultDataLoader
from data.Cifar10FlipShiftDataLoader import Cifar10FlipShiftDataLoader
from network.LayerInput import LayerInput
from network.LayerOutput import LayerOutput
from network.LayerDropout import LayerDropout
from network.LayerFC import LayerFC
from network.LayerBatchnorm import LayerBatchnorm
from network.LayerActivationTanh import LayerActivationTanh
from network.LayerConv import LayerConv
from network.LayerPooling import LayerPooling
from network.LayerFlatten import LayerFlatten
from network.LayerLocalReparameterization import LayerLocalReparameterization
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


def getInitialParametersFromDict(p_dict, layer_idx, layer_type, prefix=''):
    '''
    This function is basically the inverse operation of addParametersToDict. Layer objects that have parameters require
    a dict where the keys do not contain the layer indices. This function strips off the layer indices and returns a
    dict only containing the parameters of a certain layer.
    '''
    if p_dict is None:
        return None
    if layer_type == 'linearforward':
        p_names = ['W', 'b',
                   'W_mu', 'b_mu',
                   'W_rho', 'b_rho',
                   'W_sigma_rho', 'b_sigma_rho',
                   'W_rhoA', 'b_rhoA',
                   'W_rhoB', 'b_rhoB',
                   'W_rhoC', 'b_rhoC',
                   'W_rhoD', 'b_rhoD',
                   'scale_factor_rho']
    elif layer_type == 'batchnorm':
        p_names = ['beta', 'gamma',
                   'avg_batch_mean', 'avg_batch_inv_std']
    else:
        raise NotImplementedError()
    # Fetch parameters from dict and strip off the layer index from the key
    p_dict_layer = {}
    for p_name in p_names:
        p_name_layer = '%s%s_%d' % (prefix, p_name, layer_idx)
        if p_name_layer in p_dict:
            print 'Loading parameter \'%s\'' % (p_name_layer)
            p_dict_layer[p_name] = p_dict[p_name_layer]
    return p_dict_layer


def getCifar100Model(init_p_dict, rng, srng):
    '''
    Constructs a CNN model with ternary weights and tanh activation function.

    @param init_p_dict: Dict containing the initial parameters.
    @param rng: Numpy rng object
    @param srng: rng object used by theano
    @return layer: The CNN model
    @return p_dict: Dictionary containing the shared variables with the model parameters. This helps to easily store
        parameters to a file during/after training
    '''
    regularization_weight = 1e-10
    weight_type = 'ternary'
    p_dict = {}

    # Layer 1: 128C3
    layer = LayerInput((3,32,32))
    layer = LayerConv(layer, 128, (3,3), (1,1), 'half', rng, srng,
                      weight_type=weight_type,
                      weight_parameterization='logits',
                      weight_initialization_method='probability',
                      regularizer='shayer',
                      regularizer_weight=regularization_weight,
                      logit_bounds=(-5., 5.),
                      initial_parameters=getInitialParametersFromDict(init_p_dict, 0, 'linearforward'))
    p_dict = addParametersToDict(layer, 0, p_dict)
    layer = LayerLocalReparameterization(layer, srng)
    layer = LayerBatchnorm(layer, alpha=0.1, average_statistics_over_predictions=True,
                           initial_parameters=getInitialParametersFromDict(init_p_dict, 0, 'batchnorm'))
    p_dict = addParametersToDict(layer, 0, p_dict)
    layer = LayerActivationTanh(layer)

    # Layer 2: 128C3-P2
    layer = LayerDropout(layer, 0.2, srng)
    layer = LayerConv(layer, 128, (3,3), (1,1), 'half', rng, srng,
                      weight_type=weight_type,
                      weight_parameterization='logits',
                      weight_initialization_method='probability',
                      regularizer='shayer',
                      regularizer_weight=regularization_weight,
                      logit_bounds=(-5., 5.),
                      initial_parameters=getInitialParametersFromDict(init_p_dict, 1, 'linearforward'))
    p_dict = addParametersToDict(layer, 1, p_dict)
    layer = LayerLocalReparameterization(layer, srng)
    layer = LayerPooling(layer, (2,2), mode='max')
    layer = LayerBatchnorm(layer, alpha=0.1, average_statistics_over_predictions=True,
                           initial_parameters=getInitialParametersFromDict(init_p_dict, 1, 'batchnorm'))
    p_dict = addParametersToDict(layer, 1, p_dict)
    layer = LayerActivationTanh(layer)

    # Layer 3: 256C3
    layer = LayerDropout(layer, 0.2, srng)
    layer = LayerConv(layer, 256, (3,3), (1,1), 'half', rng, srng,
                      weight_type=weight_type,
                      weight_parameterization='logits',
                      weight_initialization_method='probability',
                      regularizer='shayer',
                      regularizer_weight=regularization_weight,
                      logit_bounds=(-5., 5.),
                      initial_parameters=getInitialParametersFromDict(init_p_dict, 2, 'linearforward'))
    p_dict = addParametersToDict(layer, 2, p_dict)
    layer = LayerLocalReparameterization(layer, srng)
    layer = LayerBatchnorm(layer, alpha=0.1, average_statistics_over_predictions=True,
                           initial_parameters=getInitialParametersFromDict(init_p_dict, 2, 'batchnorm'))
    p_dict = addParametersToDict(layer, 2, p_dict)
    layer = LayerActivationTanh(layer)

    # Layer 4: 256C3-P2
    layer = LayerDropout(layer, 0.3, srng)
    layer = LayerConv(layer, 256, (3,3), (1,1), 'half', rng, srng,
                      weight_type=weight_type,
                      weight_parameterization='logits',
                      weight_initialization_method='probability',
                      regularizer='shayer',
                      regularizer_weight=regularization_weight,
                      logit_bounds=(-5., 5.),
                      initial_parameters=getInitialParametersFromDict(init_p_dict, 3, 'linearforward'))
    p_dict = addParametersToDict(layer, 3, p_dict)
    layer = LayerLocalReparameterization(layer, srng)
    layer = LayerPooling(layer, (2,2), mode='max')
    layer = LayerBatchnorm(layer, alpha=0.1, average_statistics_over_predictions=True,
                           initial_parameters=getInitialParametersFromDict(init_p_dict, 3, 'batchnorm'))
    p_dict = addParametersToDict(layer, 3, p_dict)
    layer = LayerActivationTanh(layer)

    # Layer 5: 512C3
    layer = LayerDropout(layer, 0.3, srng)
    layer = LayerConv(layer, 512, (3,3), (1,1), 'half', rng, srng,
                      weight_type=weight_type,
                      weight_parameterization='logits',
                      weight_initialization_method='probability',
                      regularizer='shayer',
                      regularizer_weight=regularization_weight,
                      logit_bounds=(-5., 5.),
                      initial_parameters=getInitialParametersFromDict(init_p_dict, 4, 'linearforward'))
    p_dict = addParametersToDict(layer, 4, p_dict)
    layer = LayerLocalReparameterization(layer, srng)
    layer = LayerBatchnorm(layer, alpha=0.1, average_statistics_over_predictions=True,
                           initial_parameters=getInitialParametersFromDict(init_p_dict, 4, 'batchnorm'))
    p_dict = addParametersToDict(layer, 4, p_dict)
    layer = LayerActivationTanh(layer)

    # Layer 6: 512C3-P2
    layer = LayerDropout(layer, 0.3, srng)
    layer = LayerConv(layer, 512, (3,3), (1,1), 'half', rng, srng,
                      weight_type=weight_type,
                      weight_parameterization='logits',
                      weight_initialization_method='probability',
                      regularizer='shayer',
                      regularizer_weight=regularization_weight,
                      logit_bounds=(-5., 5.),
                      initial_parameters=getInitialParametersFromDict(init_p_dict, 5, 'linearforward'))
    p_dict = addParametersToDict(layer, 5, p_dict)
    layer = LayerLocalReparameterization(layer, srng)
    layer = LayerPooling(layer, (2,2), mode='max')
    layer = LayerBatchnorm(layer, alpha=0.1, average_statistics_over_predictions=True,
                           initial_parameters=getInitialParametersFromDict(init_p_dict, 5, 'batchnorm'))
    p_dict = addParametersToDict(layer, 5, p_dict)
    layer = LayerActivationTanh(layer)
    layer = LayerFlatten(layer)

    # Layer 7: FC1024
    layer = LayerDropout(layer, 0.4, srng)
    layer = LayerFC(layer, 1024, rng, srng,
                    weight_type=weight_type,
                    weight_parameterization='logits',
                    weight_initialization_method='probability',
                    regularizer='shayer',
                    regularizer_weight=regularization_weight,
                    logit_bounds=(-5., 5.),
                    initial_parameters=getInitialParametersFromDict(init_p_dict, 6, 'linearforward'))
    p_dict = addParametersToDict(layer, 6, p_dict)
    layer = LayerLocalReparameterization(layer, srng)
    layer = LayerBatchnorm(layer, alpha=0.1, average_statistics_over_predictions=True,
                           initial_parameters=getInitialParametersFromDict(init_p_dict, 6, 'batchnorm'))
    p_dict = addParametersToDict(layer, 6, p_dict)
    layer = LayerActivationTanh(layer)

    # Layer 8: FC10
    layer = LayerFC(layer, 100, rng, srng,
                    weight_type=weight_type,
                    weight_parameterization='logits',
                    weight_initialization_method='probability',
                    regularizer='shayer',
                    regularizer_weight=regularization_weight,
                    enable_bias=True,
                    bias_type='real',
                    enable_activation_normalization=True,
                    logit_bounds=(-5., 5.),
                    initial_parameters=getInitialParametersFromDict(init_p_dict, 7, 'linearforward'))
    p_dict = addParametersToDict(layer, 7, p_dict)
    layer = LayerLocalReparameterization(layer, srng)
    layer = LayerOutput(layer, 100, objective='crossentropy')

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
    dataset_file = 'cifar100.npz'
    if not isfile(dataset_file):
        downloadCifar100(dataset_file)
    
    data = dict(np.load(dataset_file))
    data['x_tr'] = ((data['x_tr'] / 255.0 * 2.0) - 1.0).astype(np.float32).reshape(-1, 3, 32, 32)
    data['x_va'] = ((data['x_va'] / 255.0 * 2.0) - 1.0).astype(np.float32).reshape(-1, 3, 32, 32)
    data['x_te'] = ((data['x_te'] / 255.0 * 2.0) - 1.0).astype(np.float32).reshape(-1, 3, 32, 32)
    data['t_tr'] = data['t_tr'].astype(np.int32)
    data['t_va'] = data['t_va'].astype(np.int32)
    data['t_te'] = data['t_te'].astype(np.int32)

    rng = np.random.RandomState()
    srng = RandomStreamsGPU(rng.randint(1, 2147462579, size=(6,)))

    # Setup data loaders
    train_generator = Cifar10FlipShiftDataLoader(data['x_tr'], data['t_tr'], 100,
            flip_axis=1, max_shift=4, requires_train=True, rng=rng)
    validation_generator = DefaultDataLoader(data['x_va'], data['t_va'], 100)
    test_generator = DefaultDataLoader(data['x_te'], data['t_te'], 100)
    
    # Load real-valued model parameters for initialization
    init_model_file = 'cifar100_model_real.npz'
    if isfile(init_model_file):
        initial_parameters = dict(np.load(init_model_file))
        print 'Loading initial parameters from \'%s\'' % (init_model_file)
        print 'Parameters:', [e for e in initial_parameters]
    else:
        raise Exception('Cannot find initial model \'%s\'' % (init_model_file))

    # Create model
    global parameters
    layer, parameters = getCifar100Model(initial_parameters, rng, srng)

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
        step_size_discrete=1e-2,
        step_size_scale_fn={'type'     : 'plateau',
                            'monitor'  : 'ce_va',
                            'cooldown' : 30,
                            'patience' : 10,
                            'factor'   : 0.5},
        n_epochs=300,
        do_bn_updates_after_epoch=True,
        callback_validation_error_decreased=[(cbErrVaDecreased, [])])

    # Store model parameters. The model parameters of the best model according to the validation error are now in
    # p_vals.
    model_file = 'cifar100_model_ternary_tanh.npz'
    print 'Optimization finished. Storing model parameters to ''%s''' % model_file
    np.savez_compressed(model_file, **p_vals)


if __name__ == '__main__':
    main()
