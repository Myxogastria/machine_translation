import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# n_text = 5000
# max_vocabulary_en = 1280
# max_vocabulary_de = 2200
# epochs = 300
# epochs_translation = 50

n_text = 10000
max_vocabulary_en = 2200
max_vocabulary_de = 3600
epochs = 100
epochs_translation = 30

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

model_en.predict_next_word('<\s>')


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

i = 3000
texts_en[i]
de_en.predict_word_count(texts_en[i], length=20, ref=texts_de[i])
de_en.predict_word_count('i have an apple.', length=20, ref='ich habe einen apfel.')
