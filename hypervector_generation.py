'''
created by Jonas Schmidt on 2/8/2023
'''

import math
import random


# rotate given vector vec by rot_amt
def rot(vec, rot_amt):
    vec_size = len(vec)

    rot_vec = vec[rot_amt % vec_size : vec_size]
    rot_vec.extend(vec[0 : rot_amt % vec_size])
    return rot_vec


# TODO: encodes a dictionary of "n-grams" across a given dictionary lexicon lex
def encode_n_grams(lex, n_grams):
    return 0


# hyperparameters
hypervector_size = 1_000
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
'''
for letter in alphabet:
    print(letter, ":", alphabet[letter])

print("ratio of all calculated 1's to -1's: ", ratio_track / (len(alphabet) * hypervector_size))
'''
# ...

### generating hypervectors for n-grams
n_grams = {}

# TODO: filter out punctuation, etc.
sentence = input("Enter a sentence to decompose into n-grams: ").lower().replace(" ", "#")

# generate a dictionary of n-grams based on input sentence
for s in range(len(sentence) - n_gram_len + 1):
    curr_gram = sentence[s:s + n_gram_len]
    n_grams[curr_gram] = []

print(n_grams)

