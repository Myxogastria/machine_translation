import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from utils import load_dataset

max_word_de = 2000
max_word_en = 2000
max_sentence_de = 1
max_sentence_en = 1

texts_en, texts_de = load_dataset('deu-eng')

class LanguageModel:
    def __init__(self, max_word):
        self.max_word = max_word
    
    def train(self, texts):
        self.word2id = {'<s>':0, '<\s>':1, '<u>':2}
        self.id2word = {0:'<s>', 1:'<\s>', 2:'<u>'}
        self.unigram = list()

        for text in texts:
            words = re.split('(\W)', text)
            id_before = self.word2id['<s>']
            id_after = self.word2id['<\s>']
            for word in words:
                if word not in [' ', '']:
                    if word not in self.word2id:
                        if len(self.word2id) >= self.max_word:
                            id_after = self.word2id['<u>']
                        else:
                            id_after = len(self.word2id)
                            self.word2id[word] = id_after
                            self.id2word[id_after] = word
                        self.unigram.append((id_before, id_after))
                    else:
                        id_after = self.word2id[word]
                        self.unigram.append((id_before, id_after))
                    id_before = id_after
            else:
                if id_after != self.word2id['<\s>']:
                    self.unigram.append((id_after, self.word2id['<\s>']))

        self.model = Sequential([
            Dense(self.max_word, input_shape=(self.max_word, ), activation='softmax', use_bias=False)
        ])

        self.model.compile(loss='categorical_crossentropy')
        self.model.summary()

        train = np.array(self.unigram)
        train_input = keras.utils.to_categorical(train[:, 0])
        train_output = keras.utils.to_categorical(train[:, 1])

        self.history = self.model.fit(train_input, train_output, 
            batch_size=1000, epochs=100)

    def predict_next_word(self, word, top=10):
        if word in self.word2id:
            id = self.word2id[word]
        else:
            id = self.word2id['<u>']
        input_vector = np.zeros((1, 2000))
        input_vector[0, id] = 1
        output = self.model.predict(input_vector)[0]

        count = top
        for i in output.argsort():
            print('{}: {}'.format(self.id2word[i], output[i]))
            count -= 1
            if count <= 0:
                return
        return

model_en = LanguageModel(max_word_en)
model_en.train(texts_en[:10000])

pd.DataFrame({'loss': model_en.history.history['loss']}).plot()
plt.yscale('log')
plt.grid()

plt.show(block=False)

model_en.predict_next_word('<u>')


model_de = LanguageModel(max_word_de)
model_de.train(texts_de[:10000])

pd.DataFrame({'loss': model_de.history.history['loss']}).plot()
plt.yscale('log')
plt.grid()

plt.show(block=False)

model_de.predict_next_word(',')


