from theano import *
import theano.tensor as T
import numpy

#def Adam(cost, params, lr=0.001, b1=0.9, b2=0.999, e=1e-8,
#         gamma=1-1e-8):
def updatesAdam(params, all_grads, lr=0.001, b1=0.9, b2=0.999, e=1e-8,
         gamma=1-1e-8, lr_scaling=1.):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]
    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf
    """
    #all_grads = theano.grad(cost, params) # This is now a parameter
    if isinstance(lr, list):
        # List contains individual learning rates for each parameter
        assert len(lr) == len(params)
    else:
        # A single learning rate is used for all parameters
        lr = [lr for _ in range(len(params))]
    t = theano.shared(numpy.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    updates = []
    for theta_previous, g, alpha in zip(params, all_grads, lr):
        m_previous = theano.shared(numpy.zeros(theta_previous.get_value().shape,
                                               dtype=theano.config.floatX))
        v_previous = theano.shared(numpy.zeros(theta_previous.get_value().shape,
                                               dtype=theano.config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - ((alpha * lr_scaling) * m_hat) / (T.sqrt(v_hat) + e) # (Update parameters)

        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))
    return updates
