from theano import *
import theano.tensor as T
import numpy

def updatesRmsprop(params, all_grads, lr=0.001, gamma=0.9, e=1e-8, lr_scaling=1.):
    """
    RMSPROP update rules
    Code implementated according to:
    http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop
    """
    if isinstance(lr, list):
        # List contains individual learning rates for each parameter
        assert len(lr) == len(params)
    else:
        # A single learning rate is used for all parameters
        lr = [lr for _ in range(len(params))]

    updates = []
    for theta_previous, g, alpha in zip(params, all_grads, lr):
        g2_previous = theano.shared(numpy.zeros(theta_previous.get_value().shape,
                                                dtype=theano.config.floatX))
        g2 = gamma * g2_previous + (1 - gamma) * g ** 2

        theta = theta_previous - (alpha * lr_scaling) * g / (T.sqrt(g2) + e)

        updates.append((g2_previous, g2))
        updates.append((theta_previous, theta))
    return updates
