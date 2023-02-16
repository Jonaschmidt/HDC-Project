'''
created by Jonas Schmidt on 2/8/2023
'''

from numba import jit, cuda
import math
import random
import numpy as np
import re


# generate a hyper vector to assign to each dictionary entry of a symbol-space
# e.g., generate a hypervector for each entry of the alphabet if symbol_space is the alphabet
# @jit(target='GPU')
def generate_hypervectors(symbol_space, hypervector_size):
    for symbol in symbol_space:
        hypervector = []
        for d in range(hypervector_size):
            ran = random.random()
            # 50/50 append a 1 or -1 to hyper_vector:
            hypervector.append(-1 + 2 * math.floor(ran + 0.5))

        symbol_space.update({symbol: hypervector})
    return symbol_space


# rotate given vector vec by rot_amt to the left
# e.g., rot([1,2,3,4], 2) returns [3,4,1,2]
# @jit(target='GPU')
def rot(vec, rot_amt):
    vec_size = len(vec)

    rot_vec = vec[rot_amt % vec_size: vec_size]
    rot_vec.extend(vec[0: rot_amt % vec_size])
    return rot_vec


# encode all elements of n_grams across symbol_space
# (ex. rrT + rH + E, where r represents a rotation operation and T,H,E are elements of an n-gram)
# @jit(target='GPU')
def encode_n_grams(symbol_space, n_grams, n_gram_len):
    # for each n-gram in the n_grams dictionary
    for n in n_grams:
        # create / clear a list, mult_vecs, of vectors to multiply
        mult_vecs = []
        # (per each n-gram) for each "letter" of the n-gram
        for i in range(n_gram_len):
            # find the lex hypervector representation of the letter,
            # rotate the vector based on its position in the n-gram,
            # append this rotated vector to mult_vecs
            mult_vecs.append(np.array(rot(symbol_space[n[i]], n_gram_len - i - 1)))

        # for each hypervector in mult_vecs, produce a Hadamard product,
        # define this n-gram by this Hadamard product
        # (note, len(mult_vecs) is equivalent to n_gram_len in this line)
        for j in range(len(mult_vecs) - 1):
            mult_vecs[j + 1] = np.multiply(mult_vecs[j], mult_vecs[j + 1])

        n_grams[n] = list(mult_vecs[-1])
        mult_vecs.clear()


# scrubs a given string as per rules described below:
'''
input scrubbing rules:
replace space characters and punctuation with "#", 
replace resulting duplicate hashmarks ("##", "###", "####", etc.) with single ("#")
'''


# TODO: improve duplicate hash replacement with regex
# @jit(target='GPU')
def scrub(sentence):
    sentence = re.sub(r'[^\w\s]', "#", sentence)
    sentence = sentence.replace(" ", "#")
    sentence = sentence.replace("###", "#")
    return sentence.replace("##", "#")

  
