'''
created by Jonas Schmidt on 3/31/2023
'''

'''
This program takes language data from the tatoeba dataset of English and Turkish sentences.
The algorithm then trains on this data.
The program then classifies the languages and outputs its test accuracy.
'''

from timeit import default_timer

start = default_timer()

print("importing...")

import tensorflow as tf
import hypervector_generation_tf as hgen
import vector_comparison_tf as vcomp
import tatoeba_access as ta

# hyperparameters
hypervector_size = 1_000
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
alphabet = hgen.generate_orthogonal_hypervectors(alphabet, hypervector_size)

# initialize class hypervectors as zero vectors
ENG_CLASS = tf.zeros(hypervector_size)
TUR_CLASS = tf.zeros(hypervector_size)

# load training and testing data
train_set, test_set = ta.load_data(train_num, test_num)

# train
print("training...", end=' ')

for i in range(train_num):
    print('#' * (-1 * (i % (train_num // 10)) + 1), end='')

    curr_label = train_set[i][0]
    curr_seq = train_set[i][1]

    n_gram_list = hgen.decompose_string(curr_seq, n_gram_len)

    acc = tf.zeros(hypervector_size)
    tf.cast(acc, dtype="float32")

    for i in n_gram_list:
        acc = hgen.sum_vec([acc, tf.cast(hgen.encode_n_gram(alphabet, i), dtype="float32")])

    if curr_label == "tur":
        TUR_CLASS = hgen.sum_vec([acc, TUR_CLASS])
    else:
        ENG_CLASS = hgen.sum_vec([acc, ENG_CLASS])

vcomp.binarize(TUR_CLASS)
vcomp.binarize(ENG_CLASS)

# test
print("\ntesting...", end="  ")

accu = 0

for seq in enumerate(test_set):
    print('#' * (-1 * (seq[0] % (test_num // 10)) + 1), end = '')

    prediction = "tur"

    n_gram_list = hgen.decompose_string(seq[1][1], n_gram_len)

    test_hypervector = tf.zeros(hypervector_size)

    for i in n_gram_list:
        test_hypervector = hgen.sum_vec([test_hypervector, tf.cast(hgen.encode_n_gram(alphabet, i), dtype="float32")])

    if vcomp.cosine_similarity(test_hypervector, TUR_CLASS) < vcomp.cosine_similarity(test_hypervector, ENG_CLASS):
        prediction = "eng"

    if prediction == seq[1][0]:
        accu += 1

print("\nAccuracy (%):", 100 * accu / test_num)

print("Time: ", default_timer() - start, "s", sep='')

