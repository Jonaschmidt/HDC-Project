'''
created by Jonas Schmidt on 3/31/2023
'''
import tensorflow as tf
import unicodedata
import re


# generate orthogonal seed vectors (1-D tensors) to assign to each dictionary entry of a symbol-space
# e.g., generate a hypervector for each letter of the alphabet if symbol_space is the alphabet
def generate_orthogonal_hypervectors(symbol_space, hypervector_size):
    for symbol in symbol_space:
        random_tensor = tf.random.uniform(shape=[hypervector_size],
                                          minval=0,
                                          maxval=2,
                                          dtype=tf.dtypes.int32,
                                          seed=None,
                                          name=None)

        symbol_space.update({symbol : tf.where(random_tensor > 0, tf.ones(shape=[hypervector_size]),
                                              -1 * tf.ones(shape=[hypervector_size]))})
    return symbol_space


# generate correlated seed vectors of a dictionary such that each symbol gets a number of positive ones per it's position
# i.e., the first entry will have a correlated vector with all -1's, the last entry vice versa
def generate_correlated_hypervectors(symbol_space, hypervector_size):
    for symbol in enumerate(symbol_space):
        # define the number of 1's to insert
        num_ones = int(hypervector_size * symbol[0] / (len(symbol_space) - 1))

        # create a vector full of -1's
        corr_vec = tf.constant([-1] * hypervector_size)

        # choose random indices to insert 1's
        indices = tf.random.shuffle(tf.range(hypervector_size))[:num_ones]

        # replace the values at the chosen indices with 1's
        updates = tf.ones(num_ones, dtype=tf.int32)
        corr_vec = tf.tensor_scatter_nd_update(corr_vec, tf.expand_dims(indices, 1), updates)

        symbol_space.update({symbol[1] : corr_vec})
    return symbol_space


# rotate given vector vec by rot_amt to the left
# e.g., rot([1,2,3,4], 2) returns [3,4,1,2]
def rot(vec, rot_amt):
    return tf.roll(input=vec, shift=-1 * rot_amt, axis=0)


# sums vectors of a given list in columnar fashion
def sum_vec(vec_list):
    return tf.reduce_sum(tf.concat([vec_list], 1), 0)


# generate a list of n-grams based on input sentence,
# returns a tensor of strings
def decompose_string(string, n_gram_len):
    return tf.strings.ngrams(tf.constant([*string]), 4)


# scrubs a given string as per rules described,
# returns a string
def scrub_string(string, default_char='#'):
    '''
    scrubbing rules:
    replace space characters and punctuation with default_character ('#' by default),
    replace resulting duplicate default_character with single default_character
    '''
    string = string.replace(" ", default_char)
    string = re.sub(r'[\d-]', default_char, string)
    string = re.sub(r'[^\w\s]+', default_char, string).lower()
    string = unicodedata.normalize('NFKD', string).encode('ASCII', 'ignore').decode('UTF-8')

    return string


# encode an n_gram
# returns a hypervector as a list
# (e.g., rrT * rH * E, where r represents a rotation operation and T,H,E are elements of an n-gram)
def encode_n_gram(symbol_space, n_gram):
    r = int(tf.strings.length(n_gram)) - 1
    mult_op1 = tf.cast(tf.ones(len(symbol_space[next(iter(symbol_space))])), tf.int32)

    n_gram = n_gram.numpy().decode()
    n_gram = n_gram.replace(" ", "")

    for sym in n_gram:
        mult_op2 = tf.cast(rot(symbol_space[sym], r), tf.int32)
        mult_op1 = tf.multiply(mult_op1, mult_op2)

        r -= 1

    ret = mult_op1

    return ret