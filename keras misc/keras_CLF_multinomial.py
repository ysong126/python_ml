from keras import models
from keras import layers

import matplotlib.pyplot as plt

import numpy as np


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels),dimension))
    for i, label in enumerate(labels):
        results[i, label] =1
    return results

# IMDB example
# binary classification

def bin_clf():
    from keras.datasets import imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)



    # a sample_size*10000 matrix that indicates word appearance of words
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences((test_data))

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    # neural net model
    model = models.Sequential()
    model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])

    # training
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]
    history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512,validation_data=(x_val,y_val))

    # history returned from fit() is an instance of History Object
    # it's member history is a dictionary. Poor naming Poor Guy
    history_dict = history.history

    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, 20+1) # shift by 1

    plt.plot(epochs,loss_values,'bo',label="Training loss")
    plt.plot(epochs,val_loss_values,'b',label='Validation loss')
    plt.title("training and validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return


# Reuters
# multiclass classification
def multinomial_clf():
    from keras.datasets import reuters
    (train_data, train_labels), (test_data,test_labels) =reuters.load_data(num_words=10000)
    len(train_data)
    len(test_data)
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])
    decoded_newswire = ' '.join(reverse_word_index.get(i-3,'?') for i in train_data[0]) # 3 words are taken /offset

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    # one hot coding
    one_hot_train_labels = to_one_hot(train_labels, dimension=46)
    one_hot_test_labels = to_one_hot(test_labels, dimension=46)

    # build the network
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(46,activation='softmax'))
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

    # validation
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]

    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]

    history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))

    # plotting the loss in training and validation
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs= range(1,len(loss)+1)

    plt.plot(epochs,loss,'bo',label='Training loss')
    plt.plot(epochs,val_loss,'b',label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # ploting the accuracy in training and validation
    plt.clf()
    acc = history.history['accuracy']  # accuracy
    val_acc= history.history['val_accuracy']  # val_accuracy

    plt.plot(epochs, acc,'bo', label = 'Training acc')
    plt.plot(epochs, val_acc, 'b', label= 'Validation acc')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # prediction
    predictions = model.predict(x_test)
    predictions[0].shape  # each is (46,) one-hot-code
    np.argmax(predictions[0])  # the index of the largest value is the class
    return