'''
created by Jonas Schmidt on 2/15/2023
'''

import hypervector_generation as hgen
import vector_comparison as vcomp

# measure execution time:
from timeit import default_timer as timer

start = timer()

# hyperparameters
hypervector_size = 10_000
n_gram_len = 3

# alphabet dictionary
alphabet = {'a':[],'b':[],'c':[],'d':[],'e':[],
            'f':[],'g':[],'h':[],'i':[],'j':[],
            'k':[],'l':[],'m':[],'n':[],'o':[],
            'p':[],'q':[],'r':[],'s':[],'t':[],
            'u':[],'v':[],'w':[],'x':[],'y':[],
            'z':[],'#':[]}
num_seed_vectors = len(alphabet)

### generating seed hypervectors for atomic elements
alphabet = hgen.generate_hypervectors(alphabet, hypervector_size)

# print statements...
for letter in alphabet:
    print(letter, ":", alphabet[letter])
# ...

### generating hypervectors for n-grams
n_grams = {}

# user input for debugging:
'''
sentence = input("Enter a sentence to decompose into n-grams: ").lower()
sentence = scrub(sentence)
'''

# test input for debugging:
sentence = ("The quick fox jumps over the lazy brown dog").lower()
sentence = hgen.scrub(sentence)

# generate a dictionary of n-grams based on input sentence
for s in range(len(sentence) - n_gram_len + 1):
    curr_gram = sentence[s:s + n_gram_len]
    n_grams[curr_gram] = []

hgen.encode_n_grams(alphabet, n_grams, n_gram_len)

# print statements...
for n in n_grams:
    print(n, ":", n_grams[n])
# ...

print("time elapsed: ", timer() - start)

