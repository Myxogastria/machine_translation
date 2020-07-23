import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.models import Model, Sequential

import re
import numpy as np

def split(sentence):
    words_split = re.split('(\W)', sentence)
    words = list()
    for word in words_split:
        if word not in [' ', '']:
            words.append(word)
    return words

class LanguageModel:
    def __init__(self, max_vocabulary):
        self.max_vocabulary = max_vocabulary
    
    def train(self, texts, epochs=100):
        self.word2id = {'<s>':0, '<\s>':1, '<u>':2}
        self.id2word = {0:'<s>', 1:'<\s>', 2:'<u>'}
        self.unigram = list()

        for text in texts:
            id_before = self.word2id['<s>']
            id_after = self.word2id['<\s>']
            for word in split(text):
                if word not in self.word2id:
                    if len(self.word2id) >= self.max_vocabulary:
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
                    self.unigram.append((self.word2id['<\s>'], self.word2id['<\s>']))

        self.model = Sequential([
            Dense(self.max_vocabulary, input_shape=(self.max_vocabulary, ),activation='softmax', use_bias=False, 
            kernel_initializer=keras.initializers.RandomNormal(seed=20200722))
        ])

        self.model.compile(loss='categorical_crossentropy')
        self.model.summary()

        train = np.array(self.unigram)
        train_input = keras.utils.to_categorical(train[:, 0], num_classes=self.max_vocabulary)
        train_output = keras.utils.to_categorical(train[:, 1], num_classes=self.max_vocabulary)

        self.history = self.model.fit(train_input, train_output, 
            batch_size=1000, epochs=epochs)
    
    def get_id(self, word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return self.word2id['<u>']

    def predict_next_word(self, word, top=10):
        if word in self.word2id:
            id = self.word2id[word]
        else:
            id = self.word2id['<u>']
        input_vector = np.zeros((1, self.max_vocabulary))
        input_vector[0, id] = 1
        output = self.model.predict(input_vector)[0]

        count = top
        for i in output.argsort()[::-1]:
            if i in self.id2word:
                print('{}: {}'.format(self.id2word[i], output[i]))
            else:
                print('KeyError({}): {}'.format(i, output[i]))
            count -= 1
            if count <= 0:
                return
        return

class TranslationModel:
    def __init__(self, model_from, model_to, max_word):
        self.model_from = model_from
        self.model_to = model_to
        self.max_word = max_word
    
    def train(self, texts_from, texts_to, epochs=100):
        train_input = list()
        train_output = list()
        for (text_from, text_to) in zip(texts_from, texts_to):
            words_from = split(text_from)
            words_to = split(text_to)

            if max(len(words_from), len(words_to)) >= self.max_word:
                continue

            train_output.append(self.make_word_count_vector(text_from))

            train_input.append(self.make_text_matrix(text_to))
        
        train_input = np.array(train_input)
        train_output = np.array(train_output).reshape((-1, self.model_from.max_vocabulary))
        print(train_input.shape)
        print(train_output.shape)
        self.model = Sequential([
            Reshape((self.model_to.max_vocabulary*self.max_word, ), input_shape=(self.model_to.max_vocabulary, self.max_word)), 
            Dense(self.model_from.max_vocabulary, 
                use_bias=True, activation='softmax')
        ])

        self.model.compile(loss='categorical_crossentropy')
        self.model.summary()
        self.history = self.model.fit(train_input, train_output, 
            batch_size=100, epochs=epochs)
    
    def make_word_count_vector(self, text_from):
        words_from = split(text_from)

        vector_output = np.zeros((self.model_from.max_vocabulary, 1))
        for word in words_from:
            vector_output[self.model_from.get_id(word)] += 1

        vector_output[self.model_from.get_id('<\s>')] += self.max_word - len(words_from)
        vector_output /= self.max_word

        return vector_output

    def make_text_matrix(self, text_to):
        ids_to = np.zeros((self.model_to.max_vocabulary, self.max_word))
        words = split(text_to)
        for i in range(self.max_word):
            if i < len(words):
                ids_to[self.model_to.get_id(words[i]), i] = 1
            else:
                ids_to[self.model_to.get_id('<\s>'), i] = 1
        return ids_to
    
    def predict_word_count(self, text_to, length=None, ref=None):
        if length is None:
            length = self.max_word

        count_vector = self.model.predict(
            np.array([self.make_text_matrix(text_to)])
        ).reshape((self.model_from.max_vocabulary, )) * self.max_word

        word_count = 0
        for i in count_vector.argsort()[::-1]:
            print('{}: {}'.format(self.model_from.id2word[i], count_vector[i]))
            word_count += np.ceil(count_vector[i])
            if word_count > length:
                break
        
        if ref is not None:
            print('')
            for word in split(ref):
                print('{}: {}'.format(word, count_vector[self.model_from.get_id(word)]))







