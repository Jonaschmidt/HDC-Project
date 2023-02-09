'''
created by Jonas Schmidt on 2/8/2023
'''


import random

# hyperparameters
hypervector_size = 1_000
n_gram_len = 3

# alphabet dictionary
alphabet = {'a':[],'b':[],'c':[],'d':[],'e':[],
            'f':[],'g':[],'h':[],'i':[],'j':[],
            'k':[],'l':[],'m':[],'n':[],'o':[],
            'p':[],'q':[],'r':[],'s':[],'t':[],
            'u':[],'v':[],'w':[],'x':[],'y':[],
            'z':[],'#':[]}

# keeps track of generated 1's to prove randomness of vector generation
ratio_track = 0

# generate a hyper vector to assign to each dictionary entry of alphabet
for letter in alphabet:
    hypervector = []
    for d in range(hypervector_size):
        ran = random.random()
        # 50/50 append a 1 or -1 to hyper_vector:
        # (could potentially be a way to optimize this)
        hypervector.append(-1)
        if ran % 0.5 == ran:
            hypervector[d] *= (-1)
            ratio_track += 1

    alphabet.update({letter: hypervector})

# print statements...
for letter in alphabet:
    print(letter, ":", alphabet[letter])

print("ratio of all calculated 1's to -1's: ", ratio_track / (len(alphabet) * hypervector_size))
# ...

