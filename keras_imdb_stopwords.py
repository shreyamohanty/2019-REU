from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

from nltk.corpus import stopwords

max_features = 500
# cut texts after this number of words (among top max_features most common words)
maxlen = 400
batch_size = 32
index_from = 3

print(max_features)
print(maxlen)

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features, index_from=index_from)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

#create dictionary for index to words
word_to_index = imdb.get_word_index()
#reverse dictionary and offset indices for index_from
index_to_word = {i+index_from:w for w,i in word_to_index.items()}
#populate missing indices
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"

#list of stopwords
stop_words = stopwords.words('english')

#remove stopwords
for j in range(len(x_train)):
    x_train[j] = [i for i in x_train[j] if index_to_word[i] not in stop_words]
for j in range(len(x_test)):
    x_test[j] = [i for i in x_test[j] if index_to_word[i] not in stop_words]

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
