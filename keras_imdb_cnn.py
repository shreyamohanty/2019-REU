from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.datasets import imdb

vocab_size = 1000
maxlen = 250
batch_size = 32 #used in KerasClassifier
epochs = 5
embedding_dim = 50

#Load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

#Pad sequences
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

#Create model function
def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen, dense1_nodes):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    '''model.add(layers.Conv1D(num_filters2, kernel_size2, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size, strides))
    model.add(layers.GlobalMaxPooling1D())'''
    model.add(layers.Flatten())
    model.add(layers.Dense(dense1_nodes, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#Parameter dictionary
param_grid = dict(num_filters=[32, 64, 128],
                  kernel_size=[5, 10, 20],
                  vocab_size=[vocab_size],
                  embedding_dim=[embedding_dim],
                  maxlen=[maxlen],
                  dense1_nodes=[10, 20, 30])

#Create wrapper
model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5, n_iter=10) #cv: number of folds in k-fold, n_iter: number of samples to test
grid_result = grid.fit(x_train, y_train)

#Evaluate testing set
test_accuracy = grid.score(x_test, y_test)

#Print results
print("Best Accuracy:", grid_result.best_score_)
print("Best Parameters:", grid_result.best_params_)
print("Test Accuracy:", test_accuracy)
