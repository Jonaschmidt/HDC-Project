'''
created by Jonas Schmidt on 2/17/2023
'''

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt


# calculate the cosine similarity between vectors a and b of equal length
# (similarity is of [-1, 1] : [opposite, equal], and 0 implies orthogonality)
def cosine_similarity(a:tf.Tensor, b:tf.Tensor):
    return (tf.tensordot(a, b, 1) / (tf.norm(a) * tf.norm(b))).numpy()


# calculate hamming distance between tensors a and b of equal length
def hamming_similarity(a:tf.Tensor, b:tf.Tensor):
    return tfa.metrics.hamming_distance(a, b).numpy()

