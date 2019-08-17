from keras.models import Sequential
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras.datasets import imdb
import random

'''Variable Initialization'''
#general variables
vocab_size = 1000
maxlen = 250
batch_size = 32
epochs = 20
embedding_dim = 50

#convolutional layer parameters
filters = [32, 64, 128, 256]
kernel_size = [2, 3, 4, 5]

#max pooling layer parameters
pool_size = [2, 3, 4, 5]
strides = None

#dense and dropout parameters
dense1_nodes = [8, 16, 32, 64, 128, 256]
dropout_rate = 0.2

'''Data Preprocessing'''
#load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

#pad sequences
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

'''Initial Population Functions'''
#layer blocks
def conv(model, temp):
    model.add(layers.Conv1D(filters=random.choice(filters), kernel_size=temp, activation='relu'))

def max_pool(model, temp):
    model.add(layers.MaxPooling1D(pool_size=temp))

def global_block(model):
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(units=random.choice(dense1_nodes), activation='relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

def flatten_block(model):
    model.add(layers.Flatten())
    model.add(layers.Dense(units=random.choice(dense1_nodes), activation='relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#create individual
def individual():
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
    index = random.choices(population=[1, 2, 3, 4], weights=[0.375, 0.25, 0.25, 0.125]) #more biased towards choosing convolutional and pooling layers than flatten
    tracker = maxlen #to make sure there aren't too many convolutional/pooling layers
    while tracker > 1:
        if index == [1]: #initializes convolutional layer with random parameters
            temp = random.choice(kernel_size)
            conv(model, temp)
            tracker = tracker//temp
        elif index == [2]: #initializes max pooling layer with random parameters
            temp = random.choice(pool_size)
            max_pool(model, temp)
            tracker = tracker//temp
        elif index == [3]: #adds global max pooling layer block and returns model
            global_block(model)
            return model
        elif index == [4]: #adds flatten layer block and returns model
            flatten_block(model)
            return model
        index = random.choices(population=[1, 2, 3, 4], weights=[0.375, 0.25, 0.25, 0.125])
    #if tracker is too small and model hasn't reached a block
    index = random.choice([3, 4])
    if index == 3:
        global_block(model)
        return model
    elif index == 4:
        flatten_block(model)
        return model

#create population
def population(count):
    return [individual() for x in range(count)]

'''Fitness Function'''
#eval fitness for a model
def fitness(model):
    check = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2) #stops training when val loss starts to increase but waits 2 epochs just in case
    best = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True) #saves the best model in case patience made it worse

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              callbacks=[check, best])

    saved_model = load_model('best_model.h5') #load best model
    score, accuracy = saved_model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    return accuracy

#calculate fitnesses for a population
def population_fitness(population):
    fitness_dict = {}
    for model in population:
        fitness_dict[model] = fitness(model)
    return fitness_dict

'''Crossover'''
def crossover(parent1, parent2): #different parents
    p1break = random.choice(range(len(parent1.layers)-4))
    p2break = random.choice(range(len(parent2.layers)-4))
    #make child1
    child1 = Sequential()
    i=0
    while i <= p1break:
        layer = parent1.layers[i]
        config = parent1.layers[i].get_config()
        layer = layers.deserialize({'class_name': layer.__class__.__name__,'config': config})
        child1.add(layer)
        i+=1
    for j in range(len(parent2.layers)):
        if j > p2break:
            layer = parent2.layers[j]
            config = parent2.layers[j].get_config()
            layer = layers.deserialize({'class_name': layer.__class__.__name__,'config': config})
            child1.add(layer)
    #make child2
    child2 = Sequential()
    i=0
    while i <= p2break:
        layer = parent2.layers[i]
        config = parent2.layers[i].get_config()
        layer = layers.deserialize({'class_name': layer.__class__.__name__,'config': config})
        child2.add(layer)
        i+=1
    for j in range(len(parent1.layers)):
        if j > p1break:
            layer = parent1.layers[j]
            cconfig = parent1.layers[j].get_config()
            layer = layers.deserialize({'class_name': layer.__class__.__name__,'config': config})
            child2.add(layer)
    return child1, child2




'''Mutation'''

'''Environmental Selection'''
