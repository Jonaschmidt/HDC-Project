'''
created by Jonas Schmidt on 2/15/2023
'''

print("IMPORTING...")

import hypervector_generation as hgen
print("hypervector_generation IMPORTED")
import hypervector_generation_tf as hgen_tf
print("hypervector_generation_tf IMPORTED")
import vector_comparison as vcomp
print("vector_comparison IMPORTED")
import vector_comparison_tf as vcomp_tf
print("vector_comparison_tf IMPORTED")
print("ALL IMPORTS COMPLETED\n")

# measure execution time:
from timeit import default_timer as timer

print("STARTING TIMER...\n")
start = timer()

# hyperparameters
hypervector_size = 10_000_000
n_gram_len = 3

# alphabet dictionary
# note that '#' character is a default
alphabet = {'a':[],'b':[],'c':[],'d':[],'e':[],
            'f':[],'g':[],'h':[],'i':[],'j':[],
            'k':[],'l':[],'m':[],'n':[],'o':[],
            'p':[],'q':[],'r':[],'s':[],'t':[],
            'u':[],'v':[],'w':[],'x':[],'y':[],
            'z':[],'#':[]}
num_seed_vectors = len(alphabet)

print("ENCODING SYMBOL-SPACE...")
### generating seed hypervectors for atomic elements
alphabet = hgen_tf.generate_hypervectors(alphabet, hypervector_size)
print("SYMBOL-SPACE ENCODED\n")

# print statements...
'''
for letter in alphabet:
    print(letter, ":", alphabet[letter])
'''
# ...

### generating hypervectors for n-grams
# initialize n_grams dictionary as an empty dictionary object
n_grams = {}

# user input for debugging:
'''
sentence = input("Enter a sentence to decompose into n-grams: ").lower()
sentence = hgen.scrub(sentence)
'''
print("SCRUBBING SENTENCE...")
# test input for debugging:
sentence = ("The quick fox jumps over the lazy brown dog").lower()
sentence = hgen_tf.scrub(sentence)
print("SENTENCE SCRUBBED:", sentence, "\n")


# generate a dictionary of n-grams based on input sentence
for s in range(len(sentence) - n_gram_len + 1):
    curr_gram = sentence[s:s + n_gram_len]
    n_grams[curr_gram] = []

print("ENCODING N-GRAMS...")
hgen_tf.encode_n_grams(alphabet, n_grams, n_gram_len)
print("N-GRAMS ENCODED\n")

# print statements...
'''
for n in n_grams:
    print(n, ":", n_grams[n])
'''
# ...

# show_vectors demonstration...
'''
vcomp.show_vectors(alphabet, dim_show=100, ones=1)
vcomp.show_vectors(n_grams, dim_show=100, ones=1)
'''
# ...

# other vcomp/vcomp_tf functions demonstration...
print("Cosine similarity of hypervectors associated with \'a\' and \'b\':", vcomp_tf.cosine_similarity(alphabet['a'], alphabet['b']), "\n")
print("Hamming distance of of hypervectors associated with \'a\' and \'b\':", vcomp_tf.hamming_similarity(alphabet['a'], alphabet['b']))
# ...

print("TIME ELAPSED:", timer() - start, "s")

