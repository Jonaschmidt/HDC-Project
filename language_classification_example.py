'''
created by Jonas Schmidt on 3/3/2023
'''
from sys import maxsize
import imdb_retriever as imdb
import numpy as np
import hypervector_generation as hgen
import vector_comparison as vcomp

#np.set_printoptions(threshold=maxsize)

file = open("WiLI-2018/x_train.txt", "r", encoding='utf-8')
train = file.readlines()
file.close()

hypervector_size = 8_000
n_gram_len = 4

train_num = 100
test_num = 50

alphabet = {'a': [], 'b': [], 'c': [], 'd': [], 'e': [],
            'f': [], 'g': [], 'h': [], 'i': [], 'j': [],
            'k': [], 'l': [], 'm': [], 'n': [], 'o': [],
            'p': [], 'q': [], 'r': [], 's': [], 't': [],
            'u': [], 'v': [], 'w': [], 'x': [], 'y': [],
            'z': [], "\n":[], '#': []}

# generate hypervectors for symbols
alphabet = hgen.generate_hypervectors(alphabet, hypervector_size)

# initialize class hypervectors as zero vectors
ENG_CLASS = np.zeros(hypervector_size)
TUR_CLASS = np.zeros(hypervector_size)

# train
print("training...")

# English:
eng = hgen.scrub(train[269])
#print(eng)

tur = hgen.scrub(train[38])
#print(tur)

eng = hgen.scrub(eng)
eng = hgen.decompose_sequence(eng, n_gram_len)

# encode the n-grams
for n in enumerate(eng):
    eng[n[0]] = hgen.encode_n_gram(alphabet, n[1])

acc = np.array(hgen.sum_vec(eng))

ENG_CLASS = hgen.sum_vec([ENG_CLASS, acc])[0]

# Turkish:
tur = hgen.scrub(tur)
tur = hgen.decompose_sequence(tur, n_gram_len)

# encode the n-grams
for n in enumerate(tur):
    tur[n[0]] = hgen.encode_n_gram(alphabet, n[1])

acc = np.array(hgen.sum_vec(tur))

TUR_CLASS = hgen.sum_vec([TUR_CLASS, acc])[0]

#print("Class hypervector for English:", ENG_CLASS)
#print("Class hypervector for Turkish:", TUR_CLASS, "\n")

print("Cosine sim:", vcomp.cosine_similarity(ENG_CLASS, TUR_CLASS))

user_in = hgen.scrub(input("Enter an English or Turkish sentence, or 'exit' to exit: "))

while user_in != 'exit':
    user_in = hgen.decompose_sequence(user_in, n_gram_len)
    user_in_vec = np.zeros(hypervector_size)

    # encode the n-grams
    for n in enumerate(user_in):
        user_in[n[0]] = hgen.encode_n_gram(alphabet, n[1])

    acc = np.array(hgen.sum_vec(user_in))

    user_in_vec = hgen.sum_vec([user_in_vec, acc])[0]

    if vcomp.cosine_similarity(ENG_CLASS, user_in_vec) > vcomp.cosine_similarity(TUR_CLASS, user_in_vec):
        prediction = "English"
    else:
        prediction = "Turkish"

    print("Prediction:", prediction, "\n")

    user_in = hgen.scrub(input("Enter an English or Turkish sentence, or 'exit' to exit: "))