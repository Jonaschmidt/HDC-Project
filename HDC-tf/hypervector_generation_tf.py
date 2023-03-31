'''
created by Jonas Schmidt on 3/31/2023
'''
import tensorflow as tf
import unicodedata
import re


# generate a seed vector (1-D tensor) to assign to each dictionary entry of a symbol-space
# e.g., generate a hypervector for each letter of the alphabet if symbol_space is the alphabet
def generate_hypervectors(symbol_space, hypervector_size):
    for symbol in symbol_space:
        random_tensor = tf.random.uniform(shape=[hypervector_size],
                                          minval=0,
                                          maxval=2,
                                          dtype=tf.dtypes.int32,
                                          seed=None,
                                          name=None)

        symbol_space.update({symbol: tf.where(random_tensor > 0, tf.ones(shape=[hypervector_size]),
                                              -1 * tf.ones(shape=[hypervector_size]))})
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


# TODO: test this for accuracy
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