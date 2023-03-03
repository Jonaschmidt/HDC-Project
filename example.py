'''
created by Jonas Schmidt on 3/3/2023
'''
import imdb_retriever as imdb
import numpy as np
import hypervector_generation as hgen
import vector_comparison as vcomp

hypervector_size = 10_000
n_gram_len = 4

train_num = 100
test_num = 100

alphabet = {'a': [], 'b': [], 'c': [], 'd': [], 'e': [],
            'f': [], 'g': [], 'h': [], 'i': [], 'j': [],
            'k': [], 'l': [], 'm': [], 'n': [], 'o': [],
            'p': [], 'q': [], 'r': [], 's': [], 't': [],
            'u': [], 'v': [], 'w': [], 'x': [], 'y': [],
            'z': [], '#': []}
num_seed_vectors = len(alphabet)

alphabet = hgen.generate_hypervectors(alphabet, hypervector_size)

train_dict = imdb.get_train(train_num)
test_dict = imdb.get_test(test_num)

POS_CLASS = np.zeros(hypervector_size)
NEG_CLASS = np.zeros(hypervector_size)

print("training...")
for sequence in enumerate(train_dict):
    sequence = sequence[1]
    seq = hgen.scrub(sequence)
    seq = hgen.decompose_sequence(seq, n_gram_len)

    for n in enumerate(seq):
        seq[n[0]] = hgen.encode_n_gram(alphabet, n[1])

    acc = np.array(sum(seq))[0]

    if train_dict[sequence] == 0:
        NEG_CLASS = hgen.sum([NEG_CLASS, acc])
    else:
        POS_CLASS = hgen.sum([POS_CLASS, acc])

POS_CLASS = hgen.binarize(POS_CLASS)
NEG_CLASS = hgen.binarize(NEG_CLASS)

correct = 0

print("testing...")
for sequence in enumerate(test_dict):
    sequence = sequence[1]
    seq = hgen.scrub(sequence)
    seq = hgen.decompose_sequence(seq, n_gram_len)

    for n in enumerate(seq):
        seq[n[0]] = hgen.encode_n_gram(alphabet, n[1])

    acc = np.array(sum(seq))[0]

    if(vcomp.cosine_similarity(NEG_CLASS, acc) > vcomp.cosine_similarity(POS_CLASS, acc)):
        prediction = 0
    else:
        prediction = 1

    actual = test_dict[sequence]
    if(prediction == actual):
        correct += 1

print("accuracy (%):", (correct / test_num) * 100)

