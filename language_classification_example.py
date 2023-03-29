'''
created by Jonas Schmidt on 3/17/2023
'''
# TODO: move binarize from hgen to vcomp
# TODO: binarize returns [[...]], change to [...]

import numpy as np
import hypervector_generation as hgen
import vector_comparison as vcomp
import tatoeba_access as ta

# hyperparameters
hypervector_size = 8_000
n_gram_len = 4
train_num = 100
test_num = 50

# define the symbol space
alphabet = {'a': [], 'b': [], 'c': [], 'd': [], 'e': [],
            'f': [], 'g': [], 'h': [], 'i': [], 'j': [],
            'k': [], 'l': [], 'm': [], 'n': [], 'o': [],
            'p': [], 'q': [], 'r': [], 's': [], 't': [],
            'u': [], 'v': [], 'w': [], 'x': [], 'y': [],
            'z': [], '\n':[], '#': []}

# generate hypervectors for symbols
alphabet = hgen.generate_hypervectors(alphabet, hypervector_size)

# initialize class hypervectors as zero vectors
ENG_CLASS = np.zeros(hypervector_size)
TUR_CLASS = np.zeros(hypervector_size)

# train
print("training...")

train_set, test_set = ta.load_data(train_num, test_num)

for i in range(train_num):
    curr_label = train_set[i][0]
    curr_seq = train_set[i][1]

    # TODO: make this an np array?
    vec = []

    # encode the n-grams
    for n in enumerate(curr_seq):
        vec.append(hgen.encode_n_gram(alphabet, n[1]))

    # accumulate the n-grams
    acc = np.array(hgen.sum_vec(vec))

    if curr_label == "tur":
        TUR_CLASS = hgen.sum_vec([acc, TUR_CLASS])
    else:
        ENG_CLASS = hgen.sum_vec([acc, ENG_CLASS])

TUR_CLASS = TUR_CLASS[0]
ENG_CLASS = ENG_CLASS[0]

print("TUR_CLASS =", TUR_CLASS)
print("ENG_CLASS =", ENG_CLASS)

hgen.binarize(TUR_CLASS)
hgen.binarize(ENG_CLASS)

print(vcomp.cosine_similarity(TUR_CLASS, ENG_CLASS))

print("testing...")

accu = 0

for seq in test_set:
    prediction = "tur"

    n_gram_list = hgen.decompose_sequence(seq[1], n_gram_len)

    test_hypervector = np.zeros(hypervector_size)

    for i in n_gram_list:
        test_hypervector = hgen.sum_vec([test_hypervector, hgen.encode_n_gram(alphabet, i)])

    if vcomp.cosine_similarity(test_hypervector, TUR_CLASS) > vcomp.cosine_similarity(test_hypervector, ENG_CLASS):
        prediction = "eng"

    if prediction == seq[0]:
        accu += 1

print("Accuracy (%):", 100 * accu / test_num)

