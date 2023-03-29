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
train_num = 1_000
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

# load training and testing data
train_set, test_set = ta.load_data(train_num, test_num)

# train
print("training...")

for i in range(train_num):
    curr_label = train_set[i][0]
    curr_seq = train_set[i][1]

    n_gram_list = hgen.decompose_sequence(curr_seq, n_gram_len)

    acc = np.zeros(hypervector_size)

    for i in n_gram_list:
        acc = hgen.sum_vec([acc, hgen.encode_n_gram(alphabet, i)])

    if curr_label == "tur":
        TUR_CLASS = hgen.sum_vec([acc, TUR_CLASS])[0]
    else:
        ENG_CLASS = hgen.sum_vec([acc, ENG_CLASS])[0]

vcomp.binarize(TUR_CLASS)
vcomp.binarize(ENG_CLASS)

# test
print("testing...")

accu = 0

for seq in test_set:
    prediction = "tur"

    n_gram_list = hgen.decompose_sequence(seq[1], n_gram_len)

    test_hypervector = np.zeros(hypervector_size)

    for i in n_gram_list:
        test_hypervector = hgen.sum_vec([test_hypervector, hgen.encode_n_gram(alphabet, i)])

    if vcomp.cosine_similarity(test_hypervector, TUR_CLASS) < vcomp.cosine_similarity(test_hypervector, ENG_CLASS):
        prediction = "eng"

    if prediction == seq[0]:
        accu += 1

print("Accuracy (%):", 100 * accu / test_num)

