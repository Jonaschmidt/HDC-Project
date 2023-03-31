'''
created by Jonas Schmidt on 3/22/23
'''

# Uses the Tatoeba dataset: https://tatoeba.org/en/downloads

import bz2
import numpy as np
import hypervector_generation_tf as hgen


# returns a tuple of train_set and test_set
def load_data(num_train, num_test):
    tur_dataset = bz2.open("Tatoeba (eng and tur)/tur_sentences.tsv.bz2", "rt", encoding='utf-8')
    eng_dataset = bz2.open("Tatoeba (eng and tur)/eng_sentences.tsv.bz2", "rt", encoding='utf-8')

    with tur_dataset as f:
        num_lines = sum(1 for _ in f)

    with eng_dataset as f:
        num_lines += sum(1 for _ in f)

    if num_train > num_lines:
        print("num_train too large, max size is", num_lines)

    if num_test > num_lines - num_train:
        print("num_test too large with num_train, max size with this num_train is", num_lines - num_train)

    ### collect train set
    train_set = np.zeros((num_train, 2), dtype=object)

    with bz2.open("Tatoeba (eng and tur)/tur_sentences.tsv.bz2", "rt", encoding='utf-8') as bz_file:
        for i in range(num_train // 2):
            seq = bz_file.readline()
            seq = hgen.scrub_string(str(seq.rstrip('\n').split('\t')))

            label = seq[1:4]
            seq = seq[5:-1]

            train_set[i, 0] = label
            train_set[i, 1] = seq

        rng = 0
        if num_train % 2 != 0:
            rng = 1

    with bz2.open("Tatoeba (eng and tur)/eng_sentences.tsv.bz2", "rt", encoding='utf-8') as bz_file:
        for i in range(num_train // 2 + rng):
            seq = bz_file.readline()
            seq = hgen.scrub_string(str(seq.rstrip('\n').split('\t')))

            label = seq[1:4]
            seq = seq[5:-1]

            i = i + num_train // 2
            train_set[i, 0] = label
            train_set[i, 1] = seq

    ### collect test set
    test_set = np.zeros((num_test, 2), dtype=object)

    with bz2.open("Tatoeba (eng and tur)/tur_sentences.tsv.bz2", "rt", encoding='utf-8') as bz_file:
        for i in range(num_test // 2):
            seq = bz_file.readline()
            seq = hgen.scrub_string(str(seq.rstrip('\n').split('\t')))

            label = seq[1:4]
            seq = seq[5:-1]

            test_set[i, 0] = label
            test_set[i, 1] = seq

        rng = 0
        if num_test % 2 != 0:
            rng = 1

    with bz2.open("Tatoeba (eng and tur)/eng_sentences.tsv.bz2", "rt", encoding='utf-8') as bz_file:
        for i in range(num_test // 2 + rng):
            seq = bz_file.readline()
            seq = hgen.scrub_string(str(seq.rstrip('\n').split('\t')))

            label = seq[1:4]
            seq = seq[5:-1]

            i = i + num_test // 2
            test_set[i, 0] = label
            test_set[i, 1] = seq

    # shuffle the datasets
    np.random.shuffle(train_set)
    np.random.shuffle(test_set)

    return train_set, test_set
