import cPickle as pkl
import numpy
import copy
import sys


import encoder
import decoder

import config

import lasagne
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX

from scipy import optimize, stats
from collections import OrderedDict, defaultdict, Counter
from numpy.random import RandomState
from scipy.linalg import norm

def story():
    z = load_models()
    z = load_inputs(z)
    #Generate a story based on a given input
    inputs = [z['cap'][a] for a in range(0, k)]

    print 'Target inputs: '
    for s in inputs[:5]:
        print s
    print ''

    # encode the inputs into skip-thought vectors
    vectors = encoder.encode(z['stv'], inputs)

    # decode the vectors back into text
    story = decoder.run_sampler(z['dec'], vectors, 50)
    print 'OUTPUT: '
    print story

def load_inputs(z):
    #Load the target inputs
    print 'Loading inputs...'
    sen = []
    with open(config.paths['inputs'], 'rb') as f:
        for line in f:
            sen.append(line.strip())
    z['cap'] = sen
    return z

def load_models():
    """
    Load the encoder and decoder
    """
    print config.paths['decmodel']

    # Decoder
    print 'Loading decoder...'
    decoder = decoder.model_load(config.paths['decodermodel'], config.paths['dictionary'])

    # Skip-thoughts
    print 'Loading encoder...'
    encoder = encoder.model_load(config.paths['vectormodels'], config.paths['vectortables'])

    # Pack up
    z = {}
    z['stv'] = encoder
    z['dec'] = decoder

    return z


