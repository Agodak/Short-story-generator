import os

import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy
import nltk

from collections import OrderedDict, defaultdict
from scipy.linalg import norm
from nltk.tokenize import word_tokenize

profile = False


def model_load(models_path, tables_path):
    #Load the model

    unimodel_path = models_path + 'uni_skip.npz'
    bimodel_path = models_path + 'bi_skip.npz'

    # Load model options
    with open('%s.pkl'%unimodel_path, 'rb') as f:
        unioptions = pkl.load(f)
    with open('%s.pkl'%bimodel_path, 'rb') as f:
        bioptions = pkl.load(f)

    # Load parameters
    uniparameters = init_parameters_uni(unioptions)
    uniparameters = load_parameters(unimodel_path, uniparameters)
    unitheanoparameters = init_theanoparameters(uniparameters)
    biparameters = init_parameters_bi(bioptions)
    biparameters = load_parameters(bimodel_path, biparameters)
    bitheanoparameters = init_theanoparameters(biparameters)

    # Extractor functions
    embedding, x_mask, ctxw2v = build_encoder_uni(unitheanoparameters, unioptions)
    f_w2v = theano.function([embedding, x_mask], ctxw2v, name='f_w2v')
    embedding, x_mask, ctxw2v = build_encoder_bi(bitheanoparameters, bioptions)
    f_w2v2 = theano.function([embedding, x_mask], ctxw2v, name='f_w2v2')

    # Tables
    utable, btable = tables_load(tables_path)

    # Store everything in a dictionary
    dictionary = {}
    dictionary['unioptions'] = unioptions
    dictionary['bioptions'] = bioptions
    dictionary['utable'] = utable
    dictionary['btable'] = btable
    dictionary['f_w2v'] = f_w2v
    dictionary['f_w2v2'] = f_w2v2

    return dictionary

def tables_load(tables_path):
    #Load the tables
    words = []
    utable = numpy.load(tables_path + 'utable.npy')
    btable = numpy.load(tables_path + 'btable.npy')
    f = open(tables_path + 'dictionary.txt', 'rb')
    for line in f:
        words.append(line.decode('utf-8').strip())
    f.close()
    utable = OrderedDict(zip(words, utable))
    btable = OrderedDict(zip(words, btable))
    return utable, btable

def encode(model, X, batch_size=128):
    #Encode inputs in the list X. Each entry will return a vector
 
    # first, do preprocessing
    X = preprocess(X)

    # word dictionary and init
    dictionary = defaultdict(lambda : 0)
    for w in model['utable'].keys():
        dictionary[w] = 1
    unifeatures = numpy.zeros((len(X), model['unioptions']['dim']), dtype='float32')
    bifeatures = numpy.zeros((len(X), 2 * model['bioptions']['dim']), dtype='float32')

    # length dictionary
    length_dict = defaultdict(list)
    inputs = [s.split() for s in X]
    for i,s in enumerate(inputs):
        length_dict[len(s)].append(i)

    # Encoding by length
    for k in length_dict.keys():
        numbatches = len(length_dict[k]) / batch_size + 1
        for minibatch in range(numbatches):
            sents = length_dict[k][minibatch::numbatches]
            uniembedding = numpy.zeros((k, len(sents), model['unioptions']['dim_word']), dtype='float32')
            biembedding = numpy.zeros((k, len(sents), model['bioptions']['dim_word']), dtype='float32')
            for ind, c in enumerate(sents):
                input = inputs[c]
                for j in range(len(input)):
                    if dictionary[input[j]] > 0:
                        uniembedding[j,ind] = model['utable'][input[j]]
                        biembedding[j,ind] = model['btable'][input[j]]
                    else:
                        uniembedding[j,ind] = model['utable']['UNK']
                        biembedding[j,ind] = model['btable']['UNK']
            uff = model['f_w2v'](uniembedding, numpy.ones((len(input),len(sents)), dtype='float32'))
            bff = model['f_w2v2'](biembedding, numpy.ones((len(input),len(sents)), dtype='float32'))
            for j in range(len(uff)):
                uff[j] /= norm(uff[j])
                bff[j] /= norm(bff[j])
            for ind, c in enumerate(sents):
                unifeatures[c] = uff[ind]
                bifeatures[c] = bff[ind]
    
    features = numpy.c_[unifeatures, bifeatures]
    return features

def preprocess(text):
    X = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for t in text:
        sents = sent_detector.tokenize(t)
        result = ''
        for s in sents:
            tokens = word_tokenize(s)
            result += ' ' + ' '.join(tokens)
        X.append(result)
    return X

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

layers = {'gru': ('param_init_gru', 'gru_layer')}

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

def init_parameters_uni(options):
    parameters = OrderedDict()

    # embedding
    parameters['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])

    # encoder: GRU
    parameters = get_layer(options['encoder'])[0](options, parameters, prefix='encoder', nin=options['dim_word'], dim=options['dim'])
    return parameters

def init_parameters_bi(options):
    parameters = OrderedDict()

    # embedding
    parameters['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])

    # encoder: GRU
    parameters = get_layer(options['encoder'])[0](options, parameters, prefix='encoder', nin=options['dim_word'], dim=options['dim'])
    parameters = get_layer(options['encoder'])[0](options, parameters, prefix='encoder_r', nin=options['dim_word'], dim=options['dim'])
    return parameters

def build_encoder_uni(theanoparameters, options):
    """
    #build encoder from word embeddings
    """
    embedding = tensor.tensor3('embedding', dtype='float32')
    x_mask = tensor.matrix('x_mask', dtype='float32')

    proj = get_layer(options['encoder'])[1](theanoparameters, embedding, options, prefix='encoder', mask=x_mask)
    ctx = proj[0][-1]

    return embedding, x_mask, ctx

def build_encoder_bi(theanoparameters, options):
    """
    build bidirectional encoder from word embeddings
    """
    embedding = tensor.tensor3('embedding', dtype='float32')
    embeddingr = embedding[::-1]
    x_mask = tensor.matrix('x_mask', dtype='float32')
    xr_mask = x_mask[::-1]

    proj = get_layer(options['encoder'])[1](theanoparameters, embedding, options, prefix='encoder', mask=x_mask)
    projr = get_layer(options['encoder'])[1](theanoparameters, embeddingr, options, prefix='encoder_r', mask=xr_mask)

    ctx = tensor.concatenate([proj[0][-1], projr[0][-1]], axis=1)

    return embedding, x_mask, ctx

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

def gru_layer(theanoparameters, state_below, options, prefix='gru', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = theanoparameters[_p(prefix,'Ux')].shape[1]

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

    rval, updates = theano.scan(_step, sequences=seqs, outputs_info = [tensor.alloc(0., n_samples, dim)], non_sequences = [theanoparameters[_p(prefix, 'U')], theanoparameters[_p(prefix, 'Ux')]], name=_p(prefix, '_layers'), n_steps=nsteps, profile=profile, strict=True)
    rval = [rval]
    return rval


