'''
created by Jonas Schmidt on 2/15/2023
'''

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt


# calculate the cosine similarity between vectors a and b of equal length
# (similarity is of [-1, 1] : [opposite, equal], and 0 implies orthogonality)
def cosine_similarity(a, b):
    if (np.count_nonzero(a) == 0 or np.count_nonzero(b) == 0):
        return 0.0
    elif (np.isnan(a).any() or np.isnan(b).any()):
        return 0.0
    else:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# calculate hamming distance between vectors a and b of equal length
def hamming_similarity(a, b):
    return hamming(a.flatten(), b.flatten())


# convert the vector to a binary vector based on the values of the vector,
# if a value positive and non-zero, it will be changed to 1, otherwise 0
def binarize(vec):
    for i, val in enumerate(vec):
        if val > 0:
            vec[i] = 1
        else:
            vec[i] = 0

    return vec


# TODO: clean up, add graph labels
def show_vectors(symbol_space, num_show=64, dim_show=64, ones=1):
    num_sentinel = 0
    dim_sentinel = 0

    x = np.arange(0, dim_show)
    y = []

    fig, ax = plt.subplots()

    # for every element of dict symbol_space up to num_show or all available...
    for sym in symbol_space:

        # ...check if trying to access outside of the symbol-space
        # if so, breaks
        if num_sentinel >= len(symbol_space):
            break

        # for each bit encoded to symbol, append to y the bit
        for bit in symbol_space[sym]:
            if dim_sentinel >= dim_show:
                break

            if bit == ones:
                y.append(bit + num_sentinel)
            else:
                y.append(-1)

            dim_sentinel += 1

        dim_sentinel = 0

        # plot each element of y over each x, then clear y for next loop execution
        ax.scatter(x[0:min(dim_show, len(y))], y, s=1)
        y.clear()

        num_sentinel += 1

        # cut off all bits != ones
        ax.set_ybound(0)

    plt.show()

