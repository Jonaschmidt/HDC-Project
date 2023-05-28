'''
created by Jonas Schmidt on 3/31/2023
'''
import tensorflow as tf


# convert the vector to a binary vector based on the values of the vector,
# if a value positive and non-zero, it will be changed to 1, otherwise 0
def binarize(in_vec):
    return tf.where(in_vec > 0, tf.ones_like(in_vec), tf.zeros_like(in_vec))


# calculate the cosine similarity between vectors a and b of equal length
# (similarity is of [-1, 1] : [opposite, equal], and 0 implies orthogonality)
def cosine_similarity(tensor_1, tensor_2):
    return (-1 * tf.keras.losses.cosine_similarity(tensor_1, tensor_2, axis=0)).numpy()
