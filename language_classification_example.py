'''
created by Jonas Schmidt on 3/17/2023
'''
import numpy as np
import hypervector_generation as hgen
import vector_comparison as vcomp

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

eng = hgen.scrub("In 1978 Johnson was awarded an American Institute of Architects Gold Medal. In 1979 he became the "
                 "first recipient of the Pritzker Architecture Prize the most prestigious international architectural"
                 " award.")
tur = hgen.scrub("Tsutinalar (İngilizce: Tsuut'ina): Kanada'da Alberta bölgesinde Calgary'de yaşarlar. Tek başına "
                 "grup oluştururlar ve Pasifik ve Güney Atabaskları ile antik yakınlıklar göstermiştir.")

### Train on English paragraph
eng = hgen.scrub(eng)
eng = hgen.decompose_sequence(eng, n_gram_len)

# encode the n-grams
for n in enumerate(eng):
    eng[n[0]] = hgen.encode_n_gram(alphabet, n[1])

# accumulate the n-grams
acc = np.array(hgen.sum_vec(eng))

ENG_CLASS = hgen.sum_vec([ENG_CLASS, acc])[0]

### Train on Turkish paragraph
tur = hgen.scrub(tur)
tur = hgen.decompose_sequence(tur, n_gram_len)

# encode the n-grams
for n in enumerate(tur):
    tur[n[0]] = hgen.encode_n_gram(alphabet, n[1])

# accumulate the n-grams
acc = np.array(hgen.sum_vec(tur))

TUR_CLASS = hgen.sum_vec([TUR_CLASS, acc])[0]

#print("Cosine sim:", vcomp.cosine_similarity(ENG_CLASS, TUR_CLASS))

user_in = hgen.scrub(input("Enter an English or Turkish sentence, or 'exit' to exit: "))

while user_in != 'exit':
    user_in = hgen.decompose_sequence(user_in, n_gram_len)
    user_in_vec = np.zeros(hypervector_size)

    # encode the n-grams
    for n in enumerate(user_in):
        user_in[n[0]] = hgen.encode_n_gram(alphabet, n[1])

    # accumulate the n-grams
    acc = np.array(hgen.sum_vec(user_in))

    user_in_vec = hgen.sum_vec([user_in_vec, acc])[0]

    if vcomp.cosine_similarity(ENG_CLASS, user_in_vec) > vcomp.cosine_similarity(TUR_CLASS, user_in_vec):
        prediction = "English"
    else:
        prediction = "Turkish"

    print("Prediction:", prediction, "\n")

    user_in = hgen.scrub(input("Enter an English or Turkish sentence, or 'exit' to exit: "))

