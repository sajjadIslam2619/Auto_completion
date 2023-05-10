import pandas as pd
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

TRAINING_MODE = True
GPU_MODE = False
TESTING_MODE = True

data = pd.read_csv('./Data/train.csv')
#print(data.head())

#print("Number of records: ", data.shape[0])
#print("Number of fields: ", data.shape[1])


data['Chief Complaint'] = data['Chief Complaint'].apply(lambda x: x.replace(u'\xa0',u' '))
data['Chief Complaint'] = data['Chief Complaint'].apply(lambda x: x.replace('\u200a',' '))


tokenizer = Tokenizer(oov_token='<oov>') # For those words which are not found in word_index
tokenizer.fit_on_texts(data['Chief Complaint'])
total_words = len(tokenizer.word_index) + 1

#print("Total number of words: ", total_words)
#print("Word: ID")
#print("------------")
#print("<oov>: ", tokenizer.word_index['<oov>'])
#print("HTN: ", tokenizer.word_index['htn'])
#print("denies: ", tokenizer.word_index['denies'])
#print("Consumption: ", tokenizer.word_index['consumption'])


input_sequences = []
for line in data['Chief Complaint']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    #print(token_list)
    
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# print(input_sequences)
print("Total input sequences: ", len(input_sequences))


# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
input_sequences[1]


# create features and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

#print(xs[5])
#print(labels[5])
#print(ys[5][14])

# Check if GPU is available
if tf.test.gpu_device_name():
    print(tf.test.gpu_device_name())
    GPU_MODE = True
    print('::::: GPU found :::::')
else:
    print("::::: No GPU found :::::")

if TRAINING_MODE == True: 
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dense(total_words, activation='softmax'))
    adam = Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # 7 epoc provides better result in validation set. around 20% accuracy. 
    if GPU_MODE:
        # Train the model on the GPU
        with tf.device('/GPU:0'):
            print('::::: Model Training with GPU :::::')
            history = model.fit(xs, ys, validation_split = 0.1, epochs=7, batch_size = 64, verbose=2)
    else: 
        print('::::: Model Training without GPU :::::')
        history = model.fit(xs, ys, validation_split = 0.1, epochs=7, batch_size = 64, verbose=2)

    print ('::::: Model Summary :::::')
    print(model.summary())
    model.save('./Trained_models/LSTM_Next_Word_CC.h5')

else:
    model = load_model('./Trained_models/LSTM_Next_Word_CC.h5')
    print ('::::: Trained Model Summary :::::')
    print(model.summary())


'''
import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
'''


# Generalized weakness
# red eye
# sent from Lake in Amandamouth for
# Has tried Imtrex and nasal
# Pelvic pain and vaginal bleeding X 1 wk. Soaking a pad about every 2 hours. Syncopal episode yesterday, denies
# Headahce, fever, chills, bodyaches, non productive

if TESTING_MODE: 
    seed_text = "Pelvic pain and vaginal bleeding X 1 wk. Soaking a pad about every 2 hours. Syncopal episode yesterday, denies"
    next_words = 2
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        # Get predicted class labels
        predicted = np.argmax(predicted, axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print('Sentence :: ', seed_text)