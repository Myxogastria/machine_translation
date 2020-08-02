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

i = 3000
translated = de_en.translate(texts_de[i], epochs=1000)
texts_de[i]
texts_en[i]
translated

pd.DataFrame({'loss': de_en.history_translation.history['loss']}).plot()
plt.yscale('log')
plt.grid()

plt.show(block=False)

# first_vector = np.zeros((de_en.model_to.max_vocabulary, 1))
# first_vector[de_en.model_to.get_id('<s>'), 0] = 1

# dummy_input = Input(shape=(de_en.model_to.max_vocabulary, de_en.model_to.max_vocabulary))
# weight = Dense(de_en.max_word, input_dim=de_en.model_to.max_vocabulary, 
#         use_bias=False, kernel_constraint=keras.constraints.MaxNorm(max_value=10000), 
#         activation=lambda x: keras.activations.softmax(x, axis=1))(dummy_input)
# weight_res = Reshape((de_en.model_to.max_vocabulary*de_en.max_word, ))(weight)
# matmul = Dense(de_en.model_from.max_vocabulary, use_bias=False, 
#     kernel_initializer=lambda shape, dtype=None: de_en.model.weights[0], 
#     bias_initializer=lambda shape, dtype=None: de_en.model.weights[1], 
#     name='translation')(weight_res)

# weight_0 = Lambda(lambda x: x[:, :, :-1], output_shape=(de_en.model_to.max_vocabulary, de_en.max_word-1))(weight)
# start_word = keras.backend.constant(first_vector, shape=(1, de_en.model_to.max_vocabulary, 1))
# weight_x = keras.layers.concatenate([start_word, weight_0], axis=2)
# weight_language = keras.backend.constant(np.array([de_en.model_to.model.weights[0].numpy()]))
# weight_y = Dot(1)([weight_language, weight_x])
# weight_y_max = Activation(lambda x: keras.activations.softmax(x, axis=1))(weight_y)
# weight_z = Multiply()([weight, weight_y_max])
# z_sum = keras.backend.sum(weight_z, axis=1)
# z_final = Lambda(lambda x: keras.backend.log(x), name='language')(z_sum)

# de_en.model_translation = Model(inputs=dummy_input, outputs=[matmul, z_final])
# de_en.model_translation.get_layer('translation').trainable = False

# dummy_data = np.array([np.eye(de_en.model_to.max_vocabulary)])

# de_en.model_translation.compile(loss={'translation':'mean_absolute_error', 'language':'mean_absolute_error'}, 
#     loss_weights={'translation':de_en.max_word**2, 'language':1}, optimizer='adam')

# answer_vec = de_en.make_word_count_vector(texts_de[i])
# de_en.history_translation = de_en.model_translation.fit(x=dummy_data, y={'translation':answer_vec.T, 'language':np.zeros((1, de_en.max_word))}, epochs=epochs)

# de_en.model_translation_x = Model(inputs=dummy_input, outputs=weight)











# model_from = lambda x: 0
# model_from.max_vocabulary = 7
# model_to = lambda x: 1
# model_to.max_vocabulary = 8
# test = lambda x: 2
# test.max_word = 8
# test.model_from = model_from
# test.model_to = model_to

# # translation_weight = np.arange(test.model_from.max_vocabulary*test.model_to.max_vocabulary*test.max_word).reshape((test.model_to.max_vocabulary*test.max_word, test.model_from.max_vocabulary))
# # language_weight = np.arange(test.model_to.max_vocabulary**2).reshape((1, test.model_to.max_vocabulary, test.model_to.max_vocabulary))
# translation_weight = np.array([
#     [1, 0, 0, 0, 0, 0, 0], 
#     [0, 1, 0, 0, 0, 0, 0], 
#     [0, 0, 0.5, 0, 0, 0, 0], 
#     [0, 0, 0.5, 0, 0, 0, 0], 
#     [0, 0, 0, 0, 1, 0, 0], 
#     [0, 0, 0, 1, 0, 0, 0], 
#     [0, 0, 0, 0, 0, 1, 1], 
#     [0, 0, 0, 0, 0, 0, 1], 
# ]*test.max_word)/test.max_word
# answer_mat = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0], 
#     [1, 0, 0, 0, 0, 0, 0, 0], 
#     [0, 1, 0, 0, 0, 0, 0, 0], 
#     [0, 0, 1, 0, 0, 0, 0, 0], 
#     [0, 0, 0, 1, 0, 0, 0, 0], 
#     [0, 0, 0, 0, 1, 0, 0, 0], 
#     [0, 0, 0, 0, 0, 1, 0, 0], 
#     [0, 0, 0, 0, 0, 0, 1, 1], 
# ])
# np.dot(answer_mat.reshape((1, 64)), translation_weight)*test.max_word
# language_weight = np.array([[
#     # [0, 0, 0, 0, 0, 0, 0, 0], 
#     # [1, 0, 0, 0, 0, 0, 0, 0], 
#     # [0, 1, 0, 0, 0, 0, 0, 0], 
#     # [0, 0, 1, 0, 0, 0, 0, 0], 
#     # [0, 0, 0, 1, 0, 0, 0, 0], 
#     # [0, 0, 0, 0, 1, 0, 0, 0], 
#     # [0, 0, 0, 0, 0, 1, 0, 0], 
#     # [0, 0, 0, 0, 0, 0, 1, 1], 
#     [0, 1, 0, 0, 0, 0, 0, 0], 
#     [0, 0, 1, 0, 0, 0, 0, 0], 
#     [0, 0, 0, 1, 0, 0, 0, 0], 
#     [0, 0, 0, 0, 1, 0, 0, 0], 
#     [0, 0, 0, 0, 0, 1, 0, 0], 
#     [0, 0, 0, 0, 0, 0, 1, 0], 
#     [0, 0, 0, 0, 0, 0, 0, 1], 
#     [0, 0, 0, 0, 0, 0, 0, 1], 
# ]])
# answer_vec = np.dot(translation_weight.T, answer_mat.reshape((64, 1)))

# first_vector = np.zeros((test.model_to.max_vocabulary, 1))
# first_vector[0, 0] = 1

# dummy_input = Input(shape=(test.model_to.max_vocabulary, test.model_to.max_vocabulary))
# weight = Dense(test.max_word, input_dim=test.model_to.max_vocabulary, 
#         # kernel_initializer=lambda shape, dtype=None: answer_mat*1000, 
#         use_bias=False, kernel_constraint=keras.constraints.MaxNorm(max_value=10000), 
#         activation=lambda x: keras.activations.softmax(x, axis=1))(dummy_input)
# weight_res = Reshape((test.model_to.max_vocabulary*test.max_word, ))(weight)
# matmul = Dense(test.model_from.max_vocabulary, use_bias=False, 
#     kernel_initializer=lambda shape, dtype=None: translation_weight, 
#     name='matmul')(weight_res)

# weight_0 = Lambda(lambda x: x[:, :, :-1], output_shape=(test.model_to.max_vocabulary, test.max_word-1))(weight)
# start_word = keras.backend.constant(first_vector, shape=(1, test.model_to.max_vocabulary, 1))
# weight_x = keras.layers.concatenate([start_word, weight_0], axis=2)
# weight_language = keras.backend.constant(language_weight*1000)
# weight_y = Dot(1)([weight_language, weight_x])
# weight_y_max = Activation(lambda x: keras.activations.softmax(x, axis=1))(weight_y)
# weight_z = Multiply()([weight, weight_y_max])
# z_sum = keras.backend.sum(weight_z, axis=1)
# z_final = Lambda(lambda x: keras.backend.log(x), name='lang')(z_sum)
# # z_sum_log = keras.backend.log(z_sum)
# # z_sum_log_sum = keras.backend.sum(z_sum_log, axis=1)
# # z_final = Lambda(lambda x: keras.backend.exp(x), name='lang')(z_sum_log_sum)

# model = Model(inputs=dummy_input, outputs=[matmul, z_final])
# model.get_layer('matmul').trainable = False

# dummy_data = np.array([np.eye(test.model_to.max_vocabulary)])
# weight_output = model.predict(dummy_data)
# weight_output

# # model.compile(loss={'matmul':'categorical_crossentropy', 'lang':'mean_absolute_error'}, 
# model.compile(loss={'matmul':'mean_absolute_error', 'lang':'mean_absolute_error'}, 
#     loss_weights={'matmul':test.max_word**2, 'lang':1}, optimizer='adam')
# # history = model.fit(x=dummy_data, y={'matmul':answer_vec.T, 'lang':np.zeros((1, 1))}, epochs=1)
# history = model.fit(x=dummy_data, y={'matmul':answer_vec.T, 'lang':np.zeros((1, test.max_word))}, epochs=10000)

# pd.DataFrame({'loss': history.history['loss']}).plot()
# plt.yscale('log')
# plt.grid()

# plt.show(block=False)

# model_x = Model(inputs=dummy_input, outputs=weight)
# a = model_x.predict(dummy_data)[0]
# model_x.predict(dummy_data)[0].argmax(0)



