import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from models import split, LanguageModel, TranslationModel
from utils import load_dataset


# n_text = 1000
# max_vocabulary_en = 400
# max_vocabulary_de = 720
# epochs = 1000
# epochs_translation = 100

# n_text = 2000
# max_vocabulary_en = 650
# max_vocabulary_de = 1150
# epochs = 500
# epochs_translation = 100

n_text = 5000
max_vocabulary_en = 1280
max_vocabulary_de = 2200
epochs = 300
epochs_translation = 50

# n_text = 10000
# max_vocabulary_en = 2200
# max_vocabulary_de = 3600
# epochs = 100
# epochs_translation = 30

texts_en, texts_de = load_dataset('deu-eng')

model_en = LanguageModel(max_vocabulary_en)
model_en.train(texts_en[:n_text], epochs=epochs)

pd.DataFrame({'loss': model_en.history.history['loss']}).plot()
plt.yscale('log')
plt.grid()

plt.show(block=False)

word_count_en = collections.Counter([model_en.id2word[id] for id in np.array(model_en.unigram).reshape((-1, ))])
len(word_count_en)
word_count_en.most_common()[:100]
word_count_en['<u>']

model_en.predict_next_word('.')


model_de = LanguageModel(max_vocabulary_de)
model_de.train(texts_de[:n_text], epochs=epochs)

pd.DataFrame({'loss': model_de.history.history['loss']}).plot()
plt.yscale('log')
plt.grid()

plt.show(block=False)

word_count_de = collections.Counter([model_de.id2word[id] for id in np.array(model_de.unigram).reshape((-1, ))])
len(word_count_de)
word_count_de.most_common()[:100]
word_count_de['<u>']

model_de.predict_next_word('<\s>')


max_de = 0
max_en = 0
for text_de, text_en in zip(texts_de[:n_text], texts_en[:n_text]):
    max_de = max(max_de, len(split(text_de)))
    max_en = max(max_en, len(split(text_en)))
print(max_de)
print(max_en)


de_en = TranslationModel(model_de, model_en, 11)
de_en.train(texts_de[:n_text], texts_en[:n_text], epochs=epochs_translation)

pd.DataFrame({'loss': de_en.history.history['loss']}).plot()
plt.yscale('log')
plt.grid()

plt.show(block=False)

i = 4000
target_text = texts_de[i]
target_text = 'ich aß einen apfel.'
target_text = 'du aßest einen apfel.'
target_text
texts_en[i]
de_en.predict_word_count(texts_en[i], length=20, ref=texts_de[i])
de_en.predict_word_count('i have an apple.', length=20, ref='ich habe einen apfel.')

n_candidate = 100

def predict_next_id(word, top=10):
    if word in de_en.model_to.word2id:
        id = de_en.model_to.word2id[word]
    else:
        id = de_en.model_to.word2id['<u>']
    input_vector = np.zeros((1, de_en.model_to.max_vocabulary))
    input_vector[0, id] = 1
    output = de_en.model_to.model.predict(input_vector)[0]

    return np.array(output.argsort()[:(-top-1):-1])

[de_en.model_to.id2word[i] for i in predict_next_id('<s>', n_candidate)]

de_en.model_to.predict_next_word('<s>', top=n_candidate)

def predict_translation_loss(text_to, ref):
    count_vector = de_en.model.predict(
        np.array([de_en.make_text_matrix(text_to)])
    ).reshape((de_en.model_from.max_vocabulary, 1))
    return np.abs(count_vector - de_en.make_word_count_vector(ref)).sum()

def get_text_to_from_id(id_list):
    text_to = ''
    for id in id_list:
        word = de_en.model_to.id2word[id]
        if word not in ['<s>', '<\s>', '<u>']:
            text_to += de_en.model_to.id2word[id] + ' '
        elif word == '<u>':
            text_to += 'unkwn'
    return text_to[:-1]

predict_translation_loss(texts_en[i], texts_de[i])

candidate = np.full((n_candidate**2, de_en.max_word), de_en.model_to.get_id('<\s>'))
candidate_loss = np.zeros((n_candidate**2, ))

test_output = np.array([de_en.make_word_count_vector(target_text)]).reshape((-1, de_en.model_from.max_vocabulary))

time0 = time.time()

k = 0
for i in predict_next_id('<s>', n_candidate):
    for j in predict_next_id(de_en.model_to.id2word[i], n_candidate):
        candidate[k][0] = i
        candidate[k][1] = j
        candidate_loss[k] = de_en.model.test_on_batch(
            np.array([de_en.make_text_matrix(get_text_to_from_id(candidate[k]))]), 
            test_output)

        # test_input.append(de_en.make_text_matrix(get_text_to_from_id(candidate[k])))
        # candidate_loss[k] = predict_translation_loss(get_text_to_from_id(candidate[k]), texts_de[i])
        k += 1

print(time.time() - time0)
print('end of {}/{}'.format(1, de_en.max_word))

for i_word in range(1, de_en.max_word):
    candidate0 = candidate
    candidate_loss0 = candidate_loss

    converged = True
    prev_vec = candidate0[candidate_loss0.argsort()[0]]
    for i in candidate_loss0.argsort()[:10]:
        converged = converged and np.all(prev_vec == candidate0[i])
        prev_vec = candidate0[i]
        print(get_text_to_from_id(candidate0[i]))
    
    if converged:
        print(time.time() - time0)
        print('end of {}/{}'.format(i_word+1, de_en.max_word))
        break

    k = 0
    for i in candidate_loss0.argsort()[:n_candidate]:
        for j in predict_next_id(de_en.model_to.id2word[candidate0[i][1]], n_candidate):
            candidate[k] = candidate0[i]
            candidate[k][2] = j
            candidate_loss[k] = de_en.model.test_on_batch(
                np.array([de_en.make_text_matrix(get_text_to_from_id(candidate[k]))]), 
                test_output)
            k += 1

    print(time.time() - time0)
    print('end of {}/{}'.format(i_word+1, de_en.max_word))

print('end')
