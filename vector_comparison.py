'''
created by Jonas Schmidt on 2/15/2023
'''

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import hamming


# calculate the cosine similarity between vectors a and b of equal length
# (similarity is of [-1, 1] : [opposite, equal], and 0 implies orthogonality)
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


# calculate hamming distance between vectors a and b of equal length
def hamming_similarity(a, b):
    return hamming(a, b)

