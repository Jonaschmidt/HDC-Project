'''
created by Jonas Schmidt on 2/8/2023
'''

import math
import random
import numpy as np
from numpy.linalg import norm
import re


# rotate given vector vec by rot_amt to the left
def rot(vec, rot_amt):
    vec_size = len(vec)

    rot_vec = vec[rot_amt % vec_size : vec_size]
    rot_vec.extend(vec[0 : rot_amt % vec_size])
    return rot_vec


# encode all elements of n_grams across lexicon lex
# (ex. rrT + rH + E, where r represents a rotation operation and T,H,E are elements of an n-gram)
def encode_n_grams(lex, n_grams):
    # for each n-gram in the n_grams dictionary
    for n in n_grams:
        # create / clear a list of vectors to multiply
        mult_vecs = []
        # (per each n-gram) for each "letter" of the n-gram
        for i in range(n_gram_len):
            # find the lex hypervector representation of the letter,
            # rotate the vector based on its position,
            # append this rotated vector to mult_vecs
            mult_vecs.append(np.array(rot(lex[n[i]], n_gram_len - i - 1)))

        # for each hypervector in mult_vecs, produce a Hadamard product,
        # define this n-gram by this Hadamard product
        # (note, len(mult_vecs) is equivalent to n_gram_len in this line)
        for j in range(len(mult_vecs) - 1):
            mult_vecs[j + 1] = np.multiply(mult_vecs[j], mult_vecs[j + 1])

        n_grams[n] = list(mult_vecs[-1])
        mult_vecs.clear()


# calculate the cosine similarity between vectors a and b of equal length
# (similarity is of [-1, 1] : [opposite, equal], and 0 implies orthogonality)
def cosine_similarity(a, b):
    return np.dot(a, b)/(norm(a) * norm(b))

# hyperparameters
hypervector_size = 100
n_gram_len = 3

# alphabet dictionary
alphabet = {'a':[],'b':[],'c':[],'d':[],'e':[],
            'f':[],'g':[],'h':[],'i':[],'j':[],
            'k':[],'l':[],'m':[],'n':[],'o':[],
            'p':[],'q':[],'r':[],'s':[],'t':[],
            'u':[],'v':[],'w':[],'x':[],'y':[],
            'z':[],'#':[]}
num_seed_vectors = len(alphabet)

# keeps track of generated 1's to prove randomness of vector generation
ratio_track = 0

### generating seed hypervectors for atomic elements
# generate a hyper vector to assign to each dictionary entry of alphabet
for letter in alphabet:
    hypervector = []
    for d in range(hypervector_size):
        ran = random.random()
        # 50/50 append a 1 or -1 to hyper_vector:
        hypervector.append(-1 + 2 * math.floor(ran + 0.5))

        ratio_track += math.floor(ran + 0.5)

    alphabet.update({letter: hypervector})

# print statements...
for letter in alphabet:
    print(letter, ":", alphabet[letter])

print("ratio of all calculated 1's to -1's: ", ratio_track / (len(alphabet) * hypervector_size))
# ...

### generating hypervectors for n-grams
n_grams = {}

# take an input and scrub it
# (replace space characters and punctuation with "#", replace any resulting duplicate hashmarks ("##", "###", "####", etc.) with single ("#"))
sentence = input("Enter a sentence to decompose into n-grams: ").lower()
sentence = re.sub(r'[^\w\s]'," ",sentence)
sentence = sentence.replace(" ", "#")
sentence = sentence.replace("###", "#")
sentence = sentence.replace("##", "#")

# generate a dictionary of n-grams based on input sentence
for s in range(len(sentence) - n_gram_len + 1):
    curr_gram = sentence[s:s + n_gram_len]
    n_grams[curr_gram] = []

encode_n_grams(alphabet, n_grams)

# print statements...
for n in n_grams:
    print(n, ":", n_grams[n])
# ...

