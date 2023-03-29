'''
created by Jonas Schmidt on 2/8/2023
'''

import math
import random
import numpy as np
import re
import unicodedata


# generate a seed vector (list) to assign to each dictionary entry of a symbol-space
# e.g., generate a hypervector for each letter of the alphabet if symbol_space is the alphabet
def generate_hypervectors(symbol_space, hypervector_size):
    for symbol in symbol_space:
        hypervector = np.ones((1, hypervector_size))
        for d in range(hypervector_size):
            ran = random.random()
            # 50/50 append a 1 or -1 to hyper_vector:
            hypervector[0][d] = (-1 + 2 * math.floor(ran + 0.5))

        symbol_space.update({symbol: hypervector})
    return symbol_space


# generate a list of n-grams based on input sentence
def decompose_sequence(sequence, n_gram_len):
    n_grams = []

    # for every "n_gram_len" symbols, append to "n_grams"
    for s in range(len(sequence) - n_gram_len + 1):
        curr_gram = sequence[s:s + n_gram_len]
        n_grams.append(curr_gram)

    return n_grams


# encode an n_gram
# returns a hypervector as a list
# (ex. rrT * rH * E, where r represents a rotation operation and T,H,E are elements of an n-gram)
def encode_n_gram(symbol_space, n_gram):
    r = len(n_gram) - 1
    mult = []

    # for each symbol in the n_gram, append the symbol-space vector to "mult"
    # after rotating the vector the appropriate amount
    for sym in n_gram:
        sym = rot(symbol_space[sym], r)
        mult.append(sym)

        r -=1

    # multiply (element-wise) each vector in "mult"
    for j in range(len(mult) - 1):
        mult[j + 1] = np.multiply(mult[j], mult[j + 1])

    # return the product of all vetors in "mult", which will be its final element
    return mult[-1]


# scrubs a given string as per rules described
def scrub(sequence, default_char='#'):
    '''
    input scrubbing rules:
    replace space characters and punctuation with default_character ('#' by default),
    replace resulting duplicate default_character with single default_character
    '''
    sequence = sequence.replace(" ", default_char)
    sequence = re.sub(r'[\d-]', default_char, sequence)
    sequence = re.sub(r'[^\w\s]+', default_char, sequence).lower()
    sequence = unicodedata.normalize('NFKD', sequence).encode('ASCII', 'ignore').decode('UTF-8')

    return sequence


# rotate given vector vec by rot_amt to the left
# e.g., rot([1,2,3,4], 2) returns [3,4,1,2]
def rot(vec, rot_amt):
    vec_size = len(vec)

    rot_vec = np.array(vec[rot_amt % vec_size: vec_size])
    rot_vec = np.concatenate((rot_vec, vec[0: rot_amt % vec_size]), axis=0)
    return rot_vec


# sums vectors in a given list
def sum_vec(vec_list):
    for i in range(len(vec_list) - 1):
        vec_list[i + 1] = np.add(vec_list[i], vec_list[i + 1])

    return vec_list[-1]

