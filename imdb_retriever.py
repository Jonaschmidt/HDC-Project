'''
created by Jonas Schmidt on 2/20/2023
see: https://keras.io/api/datasets/imdb/
'''

from tensorflow import keras

# Use the default parameters to keras.datasets.imdb.load_data
start_char = 1
oov_char = 2
index_from = 3
# Retrieve the training sequences.
(x_train, train_label), (x_test, test_label) = keras.datasets.imdb.load_data(
    start_char=start_char, oov_char=oov_char, index_from=index_from
)
# Retrieve the word index file mapping words to indices
word_index = keras.datasets.imdb.get_word_index()
# Reverse the word index to obtain a dict mapping indices to words
# And add `index_from` to indices to sync with `x_train`
inverted_word_index = dict(
    (i + index_from, word) for (word, i) in word_index.items()
)
# Update `inverted_word_index` to include `start_char` and `oov_char`
inverted_word_index[start_char] = "[START]"
inverted_word_index[oov_char] = "[OOV]"
# Decode the first sequence in the dataset


# return a dict of "num" train entries associated with a positive or negative comment
def get_train(num=1024):
    ret_dict = {}

    for i in range(num):
        ret_dict[(" ".join(inverted_word_index[i] for i in x_train[i]))] = train_label[i]

    return ret_dict


# return a dict of "num" test entries associated with a positive or negative comment
def get_test(num=1024):
    ret_dict = {}

    for i in range(num):
        ret_dict[(" ".join(inverted_word_index[i] for i in x_test[i]))] = test_label[i]

    return ret_dict

