import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy

from search import generate_sample
from collections import OrderedDict


def model_load(path_to_model, path_to_dictionary):
    """
    Load a trained model for decoding
    """
    # Load the word dictionary
    with open(path_to_dictionary, 'rb') as f:
        worddictionary = pkl.load(f)

    # Create inverted dictionary
    word_inverteddictionary = dict()
    for kk, vv in worddictionary.iteritems():
        word_inverteddictionary[vv] = kk
    word_inverteddictionary[0] = '<eos>'
    word_inverteddictionary[1] = 'UNK'

    # Load model options
    with open('%s.pkl'%path_to_model, 'rb') as f:
        options = pkl.load(f)
    if 'doutput' not in options.keys():
        options['doutput'] = True

    # Load parameters
    parameters = init_parameters(options)
    parameters = load_parameters(path_to_model, parameters)
    theanoparameters = init_theanoparameters(parameters)

    # Sampler
    random = RandomStreams(1234)
    f_init, f_next = build_sampler(theanoparameters, options, random)

    # Pack everything up
    decoder = dict()
    decoder['options'] = options
    decoder['random'] = random
    decoder['worddictionary'] = worddictionary
    decoder['word_inverteddictionary'] = word_inverteddictionary
    decoder['theanoparameters'] = theanoparameters
    decoder['f_init'] = f_init
    decoder['f_next'] = f_next
    return decoder

def run_sampler(decoder, vectors, beam_width):
    """
    Generate text conditioned on vectors
    """
    sample, score = generate_sample(decoder['theanoparameters'], decoder['f_init'], decoder['f_next'], vectors.reshape(1, -1), decoder['options'], beam_width, maxlen=1000)
    text = []
    for c in sample:
        text.append(' '.join([decoder['word_inverteddictionary'][w] for w in c[:-1]]))

    lengths = numpy.array([len(s.split()) for s in text])
    if lengths[0] == 0:
        lengths = lengths[1:]
        score = score[1:]
        text = text[1:]
    sidx = numpy.argmin(score)
    text = text[sidx]
    score = score[sidx]

    return text

def _p(pp, name):
    return '%s_%s'%(pp, name)

def init_theanoparameters(parameters):
    theanoparameters = OrderedDict()
    for kk, pp in parameters.iteritems():
        theanoparameters[kk] = theano.shared(parameters[kk], name=kk)
    return theanoparameters

def load_parameters(path, parameters):
    pp = numpy.load(path)
    for kk, vv in parameters.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive'%kk)
            continue
        parameters[kk] = pp[kk]
    return parameters

layers = {'feedforward': ('param_init_feedforwardlayer', 'feedforwardlayer'), 'gru': ('param_init_gru', 'gru_layer')}

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

def init_parameters(options):
    """
    Initialize all parameters
    """
    parameters = OrderedDict()

    # Word embeddingedding
    parameters['Wembedding'] = norm_weight(options['n_words'], options['dim_word'])

    # init state
    parameters = get_layer('feedforward')[0](options, parameters, prefix='feedforward_state', nin=options['dimctx'], nout=options['dim'])

    # Decoder
    parameters = get_layer(options['decoder'])[0](options, parameters, prefix='decoder',
                                              nin=options['dim_word'], dim=options['dim'])

    # Output layer
    if options['doutput']:
        parameters = get_layer('feedforward')[0](options, parameters, prefix='feedforward_hid', nin=options['dim'], nout=options['dim_word'])
        parameters = get_layer('feedforward')[0](options, parameters, prefix='feedforward_logit', nin=options['dim_word'], nout=options['n_words'])
    else:
        parameters = get_layer('feedforward')[0](options, parameters, prefix='feedforward_logit', nin=options['dim'], nout=options['n_words'])

    return parameters

def build_sampler(theanoparameters, options, random):
    """
    Forward sampling
    """
    ctx = tensor.matrix('ctx', dtype='float32')
    ctx0 = ctx

    init_state = get_layer('feedforward')[1](theanoparameters, ctx, options, prefix='feedforward_state', activ='tanh')
    f_init = theano.function([ctx], init_state, name='f_init', profile=False)

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, embedding should be all zero
    embedding = tensor.switch(y[:,None] < 0, tensor.alloc(0., 1, theanoparameters['Wembedding'].shape[1]), theanoparameters['Wembedding'][y])

    # decoder
    proj = get_layer(options['decoder'])[1](theanoparameters, embedding, init_state, options, prefix='decoder', mask=None, one_step=True)
    next_state = proj[0]

    # output
    if options['doutput']:
        hid = get_layer('feedforward')[1](theanoparameters, next_state, options, prefix='feedforward_hid', activ='tanh')
        logit = get_layer('feedforward')[1](theanoparameters, hid, options, prefix='feedforward_logit', activ='linear')
    else:
        logit = get_layer('feedforward')[1](theanoparameters, next_state, options, prefix='feedforward_logit', activ='linear')
    next_probs = tensor.nnet.softmax(logit)
    next_sample = random.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    inps = [y, init_state]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=False)

    return f_init, f_next

def linear(x):
    return x

def tanh(x):
    return tensor.tanh(x)

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.1, ortho=True):
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = numpy.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype('float32')

# Feedforward layer
def param_init_feedforwardlayer(options, parameters, prefix='feedforward', nin=None, nout=None, ortho=True):
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['dim_proj']
    parameters[_p(prefix,'W')] = norm_weight(nin, nout)
    parameters[_p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')

    return parameters

def feedforwardlayer(theanoparameters, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    """
    Feedforward pass
    """
    return eval(activ)(tensor.dot(state_below, theanoparameters[_p(prefix,'W')])+theanoparameters[_p(prefix,'b')])

# GRU layer
def param_init_gru(options, parameters, prefix='gru', nin=None, dim=None):
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    W = numpy.concatenate([norm_weight(nin,dim), norm_weight(nin,dim)], axis=1)
    parameters[_p(prefix,'W')] = W
    parameters[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim), ortho_weight(dim)], axis=1)
    parameters[_p(prefix,'U')] = U

    Wx = norm_weight(nin, dim)
    parameters[_p(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    parameters[_p(prefix,'Ux')] = Ux
    parameters[_p(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

    return parameters

def gru_layer(theanoparameters, state_below, init_state, options, prefix='gru', mask=None, one_step=False, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = theanoparameters[_p(prefix,'Ux')].shape[1]

    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(state_below, theanoparameters[_p(prefix, 'W')]) + theanoparameters[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, theanoparameters[_p(prefix, 'Wx')]) + theanoparameters[_p(prefix, 'bx')]
    U = theanoparameters[_p(prefix, 'U')]
    Ux = theanoparameters[_p(prefix, 'Ux')]

    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    if one_step:
        rval = _step(*(seqs+[init_state, theanoparameters[_p(prefix, 'U')], theanoparameters[_p(prefix, 'Ux')]]))
    else:
        rval, updates = theano.scan(_step, sequences=seqs, outputs_info = [init_state], non_sequences = [theanoparameters[_p(prefix, 'U')], theanoparameters[_p(prefix, 'Ux')]], name=_p(prefix, '_layers'), n_steps=nsteps, profile=False, strict=True)
    rval = [rval]
    return rval

