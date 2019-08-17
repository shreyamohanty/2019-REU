from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 500
print(max_features)
# cut texts after this number of words (among top max_features most common words)
maxlen = 200
print(maxlen)
batch_size = 32

start_char = 1
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features, start_char=start_char)

for lst in x_train:
    if len(lst) > maxlen:
        off = (len(lst) - maxlen) // 2
        for i in range(off): #remove words from end
            lst.pop()
        lst.reverse()
        for i in range(off+1): #remove words from end of reversed (so beginning)
            lst.pop()
        lst.append(start_char) #add in start char
        lst.reverse() #reverse to normal order

for lst in x_test:
    if len(lst) > maxlen:
        off = (len(lst) - maxlen) // 2
        for i in range(off): #remove words from end
            lst.pop()
        lst.reverse()
        for i in range(off+1): #remove words from end of reversed (so beginning)
            lst.pop()
        lst.append(start_char) #add in start char
        lst.reverse() #reverse to normal order

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

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