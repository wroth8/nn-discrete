import theano
import theano.tensor as T
from optimization.updatesAdam import updatesAdam
# from optimization.updatesSgdMomentum import updatesSgdMomentum
from optimization.updatesRmsprop import updatesRmsprop
from time import time
import numpy as np

def optimizeNetwork(model,
                    loader_tr=None,
                    loader_va=None,
                    loader_te=None,
                    optimization_algorithm='adam',
                    step_size=1e-3,
                    step_size_real=None,
                    step_size_discrete=None,
                    step_size_bias=None,
                    step_size_batchnorm=None,
                    step_size_scale_factor=None,
                    momentum_g2=0.99,
                    step_size_scale_fn=lambda n_epochs, n_updates : 0.99 ** n_epochs,
                    n_epochs=1000,
                    rng=None, # obsolete
                    rng_seed=1, # obsolete
                    approximate_training_error='running_average_train',
                    do_bn_updates_after_epoch=False,
                    evaluate_top5=False,
                    evaluate_tr_at_start=True,
                    callback_epoch_start=[],
                    callback_epoch_end=[],
                    callback_validation_error_decreased=[],
                    callback_optimization_finished=[],
                    enable_debug_mode_nan_updates=False,
                    enable_debug_mode_nan_activations=False):
    assert optimization_algorithm in ['adam', 'rmsprop']
    assert approximate_training_error in ['running_average_train', 'running_average_predict', False]
    
    assert callable(step_size_scale_fn) or isinstance(step_size_scale_fn, dict)
    if isinstance(step_size_scale_fn, dict):
        assert 'type' in step_size_scale_fn
        assert step_size_scale_fn['type'] in ['plateau']
        if step_size_scale_fn['type'] == 'plateau':
            assert 'monitor' in step_size_scale_fn
            assert 'patience' in step_size_scale_fn
            assert 'cooldown' in step_size_scale_fn
            assert 'factor' in step_size_scale_fn
            assert step_size_scale_fn['monitor'] in ['cost', 'ce_va', 'ce_va_top1', 'ce_va_top5']
            assert step_size_scale_fn['monitor'] != 'ce_va_top5' or evaluate_top5 == True
            lr_scale = 1.
            plateau_last_update_idx = -1
    
    step_size_real         = step_size if step_size_real is None         else step_size_real
    step_size_discrete     = step_size if step_size_discrete is None     else step_size_discrete
    step_size_bias         = step_size if step_size_bias is None         else step_size_bias
    step_size_batchnorm    = step_size if step_size_batchnorm is None    else step_size_batchnorm
    step_size_scale_factor = step_size if step_size_scale_factor is None else step_size_scale_factor
    
    x = model.getSymbolicInput()
    t = model.getSymbolicTarget()
    
    if rng == None:
        rng = np.random.RandomState(rng_seed)

    p_names_weights_real = ['W', 'W_mu', 'W_sigma_rho']
    p_names_weights_discrete = ['W_rho', 'W_rhoA', 'W_rhoB', 'W_rhoC', 'W_rhoD', 'W_rhoM1', 'W_rhoM05', 'W_rhoP05', 'W_rhoP1']
    p_names_bias = ['b', 'b_mu', 'b_rho', 'b_sigma_rho', 'b_rhoA', 'b_rhoB', 'b_rhoC', 'b_rhoD', 'b_rhoM1', 'b_rhoM05', 'b_rhoP05', 'b_rhoP1']
    p_names_batchnorm = ['beta', 'gamma']
    p_names_scale_factor = ['scale_factor_rho']
    parameters, step_size_lst = [], []
    bounded_params_idx_lst = []
    print 'Initial step sizes:'
    for p_name, p_step_size_scaler, p_is_trainable, p_bounds, p in [(p['name'], p['step_size_scaler'], p['trainable'], p['bounds'], p['param']) for p in model.getParameterEntries()]:
        if not p_is_trainable:
            continue
        
        if p_bounds is not None:
            bounded_params_idx_lst.append( (len(parameters), p_name, p_bounds) ) # len(parameters) gives the corresponding index in the list

        if p_name in p_names_weights_real:
            p_step_size = step_size_real
        elif p_name in p_names_weights_discrete:
            p_step_size = step_size_discrete
        elif p_name in p_names_bias:
            p_step_size = step_size_bias
        elif p_name in p_names_batchnorm:
            p_step_size = step_size_batchnorm
        elif p_name in p_names_scale_factor:
            p_step_size = step_size_scale_factor
        else:
            raise Exception('Unrecognized parameter: \'%s\'' % p_name)
        p_step_size = p_step_size * p_step_size_scaler
        print 'Parameter %18s: %e (%s)' % ('\'%s\'' % p_name, p_step_size, 'unbounded' if p_bounds is None else ('bounds=%s' % str(p_bounds)))
        step_size_lst.append(p_step_size)        
        parameters.append(p)

    LR_scale = T.fscalar('Learning Rate Scale')
    if optimization_algorithm == 'adam':
        updates = updatesAdam(parameters,
                              T.grad(model.getCost(), parameters),
                              lr=step_size_lst,
                              gamma=1,
                              lr_scaling=LR_scale)
        for idx, p_name, p_bounds in bounded_params_idx_lst:
            print 'Adding bound-clipping (%s) to parameter \'%s\' with idx %d' % (str(p_bounds), p_name, idx)
            updates[idx * 3 + 2] = (updates[idx * 3 + 2][0], T.clip(updates[idx * 3 + 2][1], p_bounds[0], p_bounds[1]))
    elif optimization_algorithm == 'rmsprop':
        updates = updatesRmsprop(parameters,
                                 T.grad(model.getCost(), model.getParameters()),
                                 lr=step_size_lst,
                                 gamma=momentum_g2,
                                 lr_scaling=LR_scale)
        assert len(bounded_parameters_lst) == 0 # TODO: Implement clipping here
    else:
        raise NotImplementedError()
    
    if loader_te is None:
        print 'Info: \'loader_te\' is None --> test data will not be evaluated'

    print 'Compiling training function'
    if do_bn_updates_after_epoch == False:
        # We perform the batchnorm updates during training
        bn_updates = model.getTrainUpdates()
        bn_updates_func = None
    else:
        bn_updates = []
        bn_updates_func = theano.function(inputs=[x], updates=model.getTrainUpdates())
    print 'bn_udpates:', bn_updates

    if approximate_training_error == False:
        # No outputs here
        train_function = theano.function(inputs=[x, t, LR_scale],
                                         updates=updates + model.getTrainUpdates())
    elif approximate_training_error == 'running_average_train':
        # Approximate the training error as running average over the minibatches.
        # The individual minibatch errors are computed using the training output.
        # This is typically faster than using the prediction output because the
        # training output is computed anyway during training.
        train_function = theano.function(inputs=[x, t, LR_scale],
                                         outputs=[model.getTrainClassificationCriterion(), model.getCostLikelihood()],
                                         updates=updates + bn_updates)
    elif approximate_training_error == 'running_average_predict':
        # Approximate the training error as running average over the minibatches.
        # The individual minibatch errors are computed using the prediction
        # output which requires computing a different path in the computation
        # graph. However, the approximation quality is better than using the
        # train output.
        train_function = theano.function(inputs=[x, t, LR_scale],
                                         outputs=[model.getPredictionOutput(), model.getCostLikelihood()],
                                         updates=updates + bn_updates)
    else:
        raise NotImplementedError('Unknown approximate_training_error \'%s\'' % (str(approximate_training_error)))

    print 'Compiling evaluation functions'
    func_predict = theano.function(inputs=[x], outputs=model.getPredictionOutput())

    if approximate_training_error == False or evaluate_tr_at_start == True:
        func_cost_likelihood = theano.function(inputs=[x, t], outputs=model.getCostLikelihood())
    
    # It is faster to compute the regularization-cost just once at the end
    # than to always return it. The regularization gradient can be computed
    # faster than its actual value.
    func_cost_regularize = theano.function(inputs=[], outputs=model.getCostRegularizer())
    
#     if models.enable_batch_norm:
#         print 'Compiling BatchNorm Update functions'
#         bn_updates = []
#         for layer_idx in range(models.n_layers):
#             bn_updates.append( theano.function(inputs=[], updates=[models.batch_norm_updates[layer_idx]], givens={models.x:x_tr}) )
    
    print 'Start optimization...'
    errs_tr_top1 = []
    errs_va_top1 = []
    errs_te_top1 = []
    errs_tr_top5 = []
    errs_va_top5 = []
    errs_te_top5 = []
    costs = []

    # Evaluate initial model. This is especially useful to test whether the
    # correct model was loaded or if the initialization method already gives
    # good results.
    if evaluate_tr_at_start:
        cost_reg = func_cost_regularize()
        cost_lik = evaluateLikelihood(model, loader_tr, func_cost_likelihood)
        err_top1, err_top5 = evaluateError(model, loader_tr, func_predict, evaluate_top5)
    else:
        cost_reg, cost_lik, err_top1, err_top5 = np.NaN, np.NaN, np.NaN, np.NaN
    costs.append( cost_lik + cost_reg )
    errs_tr_top1.append( err_top1 )
    errs_tr_top5.append( err_top5 )        
        
    err_top1, err_top5 = evaluateError(model, loader_va, func_predict, evaluate_top5)
    errs_va_top1.append( err_top1 )
    errs_va_top5.append( err_top5 )
    err_top1, err_top5 = evaluateError(model, loader_te, func_predict, evaluate_top5) if loader_te is not None else (np.NaN, np.NaN)
    errs_te_top1.append( err_top1 )
    errs_te_top5.append( err_top5 )

    computation_time = []

    if enable_debug_mode_nan_updates:
        debug_nan_updates = theano.function(
            inputs=[x, t, LR_scale],
            outputs=[e[0] for e in updates] + [e[1] for e in updates])
    
    if enable_debug_mode_nan_activations:
        debug_activation_names, debug_activation_vars = [], []
        for idx, (a_name, a_var) in enumerate(model.getTrainOutputsNames()):
            if isinstance(a_var, tuple):
                debug_activation_names += ['%s_mean_%d' % (a_name, idx), '%s_var_%d' % (a_name, idx)]
                debug_activation_vars += [a_var[0], a_var[1]]
            else:
                debug_activation_names += ['%s_%d' % (a_name, idx)]
                debug_activation_vars += [a_var]
        debug_nan_activations = theano.function(
            inputs=[x],
            outputs=debug_activation_vars)
    

    n_updates = 0
    best_err_va = np.Inf
    if evaluate_top5 == True:
        print 'Initial: cost=%1.5e, TOP1: ce_tr=%2.5f, ce_va=%2.5f, ce_te=%2.5f, TOP5: ce_tr=%2.5f, ce_va=%2.5f, ce_te=%2.5f' % (costs[-1], errs_tr_top1[-1], errs_va_top1[-1], errs_te_top1[-1], errs_tr_top5[-1], errs_va_top5[-1], errs_te_top5[-1])
    else:
        print 'Initial: cost=%1.5e, ce_tr=%2.5f, ce_va=%2.5f, ce_te=%2.5f' % (costs[-1], errs_tr_top1[-1], errs_va_top1[-1], errs_te_top1[-1])
    for epoch_idx in range(n_epochs):
        handleCallback(callback_epoch_start,
                       model, epoch_idx, n_updates,
                       errs_tr_top1[-1], errs_va_top1[-1], errs_te_top1[-1],
                       errs_tr_top5[-1], errs_va_top5[-1], errs_te_top5[-1],
                       costs[-1] - cost_reg, cost_reg)
        t_start = time()

        err_tr_top1, err_tr_top5, cost_lik, w_tr = [], [], [], []
        for batch_idx, (x_batch, t_batch) in enumerate(loader_tr.generateTrainData()):
            if callable(step_size_scale_fn):
                lr_scale = step_size_scale_fn(epoch_idx, epoch_idx * n_epochs + batch_idx)

            if approximate_training_error == False:
                train_function(x_batch, t_batch, lr_scale)
            else:
                res = train_function(x_batch, t_batch, lr_scale)
                if evaluate_top5 == False:
                    err_tr_top1.append( np.mean(np.argmax(res[0], axis=1) != t_batch) )
                    err_tr_top5.append( np.NaN )
                else:
                    pred = np.argsort(res[0], axis=1)[:, -5:]
                    err_tr_top1.append( np.mean(pred[:, -1] != t_batch) )
                    err_tr_top5.append( np.mean(np.all(pred != t_batch[:, None], axis=1)) )
                cost_lik.append( res[1] )
                w_tr.append( x_batch.shape[0] )
                
            n_updates += 1

        if bn_updates_func is not None:
            assert not bn_updates # bn_updates must be False here (otherwise bn_updates were already made during training)
            assert do_bn_updates_after_epoch
            for x_batch, _ in loader_tr.generateTrainData():
                bn_updates_func(x_batch)

        cost_reg = func_cost_regularize()
        if approximate_training_error == False:
            cost_lik = evaluateLikelihood(model, loader_tr, func_cost_likelihood)
            err_top1, err_top5 = evaluateError(model, loader_tr, func_predict, evaluate_top5)
            errs_tr_top1.append( err_top1 )
            errs_tr_top5.append( err_top5 )
        else:
            cost_lik = np.average(cost_lik, weights=w_tr)
            errs_tr_top1.append( np.average(err_tr_top1, weights=w_tr) )
            errs_tr_top5.append( np.average(err_tr_top5, weights=w_tr) )
        costs.append( cost_lik + cost_reg )

        err_top1, err_top5 = evaluateError(model, loader_va, func_predict, evaluate_top5)
        errs_va_top1.append( err_top1 )
        errs_va_top5.append( err_top5 )
        err_top1, err_top5 = evaluateError(model, loader_te, func_predict, evaluate_top5) if loader_te is not None else (np.NaN, np.NaN)
        errs_te_top1.append( err_top1 )
        errs_te_top5.append( err_top5 )

        t_elapsed = time() - t_start
        computation_time.append(t_elapsed)
        
        if evaluate_top5 == True:
            print 'Epoch %5d/%5d finished: cost=%1.5e, cost_logl=%7.5e, cost_reg=%7.5e, TOP1: ce_tr=%2.5f, ce_va=%2.5f, ce_te=%2.5f, TOP5: ce_tr=%2.5f, ce_va=%2.5f, ce_te=%2.5f, LR=%2.5e (%f seconds)' % (epoch_idx, n_epochs, costs[-1], cost_lik, cost_reg, errs_tr_top1[-1], errs_va_top1[-1], errs_te_top1[-1], errs_tr_top5[-1], errs_va_top5[-1], errs_te_top5[-1], step_size * lr_scale, t_elapsed)
        else:
            print 'Epoch %5d/%5d finished: cost=%1.5e, cost_logl=%7.5e, cost_reg=%7.5e, ce_tr=%2.5f, ce_va=%2.5f, ce_te=%2.5f, LR=%2.5e (%f seconds)' % (epoch_idx, n_epochs, costs[-1], cost_lik, cost_reg, errs_tr_top1[-1], errs_va_top1[-1], errs_te_top1[-1], step_size * lr_scale, t_elapsed)

        # Keep track of the best model
        if errs_va_top1[-1] < best_err_va - 1e-8:
            print 'old_err_va=%f, new_err_va=%f' % (best_err_va, errs_va_top1[-1])
            best_err_va = errs_va_top1[-1]
            handleCallback(callback_validation_error_decreased,
                           model, epoch_idx, n_updates,
                           errs_tr_top1[-1], errs_va_top1[-1], errs_te_top1[-1],
                           errs_tr_top5[-1], errs_va_top5[-1], errs_te_top5[-1],
                           costs[-1] - cost_reg, cost_reg)

        handleCallback(callback_epoch_end,
                       model, epoch_idx, n_updates,
                       errs_tr_top1[-1], errs_va_top1[-1], errs_te_top1[-1],
                       errs_tr_top5[-1], errs_va_top5[-1], errs_te_top5[-1],
                       costs[-1] - cost_reg, cost_reg)

        if isinstance(step_size_scale_fn, dict):
            if step_size_scale_fn['type'] == 'plateau':
                if step_size_scale_fn['monitor'] == 'cost':
                    plateau_best_idx = np.argmin(costs[1:]) # use [1:] since [0] contains the initial error which should be ignored
                elif step_size_scale_fn['monitor'] in ['ce_va', 'ce_va_top1']:
                    plateau_best_idx = np.argmin(errs_va_top1[1:])
                elif step_size_scale_fn['monitor'] == 'ce_va_top5':
                    plateau_best_idx = np.argmin(errs_va_top5[1:])
                else:
                    raise NotImplementedError()
                if epoch_idx - plateau_last_update_idx >= step_size_scale_fn['cooldown'] and \
                   epoch_idx - max(plateau_best_idx, plateau_last_update_idx) >= step_size_scale_fn['patience']:
                    print 'No improvement in critertion \'%s\' over the last %d iterations. Reducing step size by a factor of %f' % (step_size_scale_fn['monitor'], step_size_scale_fn['patience'], step_size_scale_fn['factor'])
                    plateau_last_update_idx = epoch_idx
                    lr_scale *= step_size_scale_fn['factor']
            else:
                raise NotImplementedError()

    handleCallback(callback_optimization_finished,
                   model, epoch_idx, n_updates,
                   errs_tr_top1[-1], errs_va_top1[-1], errs_te_top1[-1],
                   errs_tr_top5[-1], errs_va_top5[-1], errs_te_top5[-1],
                   costs[-1] - cost_reg, cost_reg)

def evaluateLikelihood(model, data_generator, func_likelihood):
    weights, likelihoods = [], []
    for x_batch, t_batch in data_generator.generateTestData():
        weights.append( x_batch.shape[0] )
        likelihoods.append( func_likelihood(x_batch, t_batch) )
    return np.average(likelihoods, weights=weights)

def evaluateError(model, data_generator, pred_func, evaluate_top5):
    weights, err_top1, err_top5 = [], [], []
    for x_batch, t_batch in data_generator.generateTestData():
        weights.append( x_batch.shape[0] )
        pred = pred_func(x_batch)
        if evaluate_top5 == False:
            err_top1.append( np.mean(np.argmax(pred, axis=1) != t_batch) )
        else:
            pred = np.argsort(pred, axis=1)[:, -5:]
            err_top1.append( np.mean(pred[:, -1] != t_batch) )
            err_top5.append( np.mean(np.all(pred != t_batch[:, None], axis=1)) )
    err_top1 = np.average(err_top1, weights=weights)
    err_top5 = np.average(err_top5, weights=weights) if evaluate_top5 == True else np.NaN 
    return err_top1, err_top5

def handleCallback(callback_lst,
                   model, n_epoch, n_updates,
                   ce_tr_top1, ce_va_top1, ce_te_top1,
                   ce_tr_top5, ce_va_top5, ce_te_top5,
                   cost_likelihood, cost_regularize):
    for callback_fn, callback_argument_lst in callback_lst:
        arg_lst = {}
        if 'model' in callback_argument_lst:
            arg_lst['model'] = model
        if 'n_epoch' in callback_argument_lst:
            arg_lst['n_epoch'] = n_epoch
        if 'n_updates' in callback_argument_lst:
            arg_lst['n_updates'] = n_updates
        if 'ce_tr' in callback_argument_lst:
            arg_lst['ce_tr'] = ce_tr_top1
        if 'ce_va' in callback_argument_lst:
            arg_lst['ce_va'] = ce_va_top1
        if 'ce_te' in callback_argument_lst:
            arg_lst['ce_te'] = ce_te_top1
        if 'ce_tr_top1' in callback_argument_lst:
            arg_lst['ce_tr_top1'] = ce_tr_top1
        if 'ce_va_top1' in callback_argument_lst:
            arg_lst['ce_va_top1'] = ce_va_top1
        if 'ce_te_top1' in callback_argument_lst:
            arg_lst['ce_te_top1'] = ce_te_top1
        if 'ce_tr_top5' in callback_argument_lst:
            arg_lst['ce_tr_top5'] = ce_tr_top5
        if 'ce_va_top5' in callback_argument_lst:
            arg_lst['ce_va_top5'] = ce_va_top5
        if 'ce_te_top5' in callback_argument_lst:
            arg_lst['ce_te_top5'] = ce_te_top5
        if 'cost_likelihood' in callback_argument_lst:
            arg_lst['cost_likelihood'] = cost_likelihood
        if 'cost_regularize' in callback_argument_lst:
            arg_lst['cost_regularize'] = cost_regularize
        callback_fn(**arg_lst)
