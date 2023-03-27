# An example from 'Deep Learning with Python'

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import kerastuner as kt
import keras.optimizers

# keras repo that holds this dataset has been updated
# dataset manually downloaded and moved to ~/.keras/datasets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# reshape datasets
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255

# re label to categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# build the model and fit  with parameter tuning using keras tuner
def model_builder(hp):
    network = models.Sequential()
    # hp - hyper parameters
    # num units
    hp_units = hp.Int('units',min_value=32,max_value=128,step=32)
    network.add(layers.Dense(units=hp_units, activation = 'relu', input_shape=(28*28,)))
    network.add(layers.Dense(10, activation = 'softmax'))

    # learning rate
    hp_learning_rate = hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4])
    network.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return network


# instantiate a Bayesian tuner
tuner=kt.Hyperband(model_builder, objective='val_accuracy', max_epochs=10, factor=3)
tuner.search(train_images, train_labels, epochs=50, validation_split=0.2)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"The hyperparameter search is complete. The optimal num of units in the 1st layer is{best_hps.get('units')}")