from keras import models
from keras import layers
from keras.datasets import boston_housing
import numpy as np
import matplotlib.pyplot as plt

# load data
(train_data, train_targets),  (test_data,test_targets) = boston_housing.load_data()

# exploratory analysis and normalization
means =  train_data.mean(axis=0)
train_data -= means
stds = train_data.std(axis=0)
train_data/= stds

test_data-=means
test_data/=stds

# build the network
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',loss='mse',metrics=['mae'])
    return model


# k fold validation
k=5
num_val_samples = len(train_data)//k
num_epoch = 100
all_scores = []
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    # pick the kth partition
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]
    # pick the rest
    partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data,partial_train_targets, validation_data=(val_data,val_targets),epochs = num_epoch,batch_size=1,verbose =0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous =  smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(all_mae_histories[10:])
plt.plot(range(1,len(smooth_mae_history)))
plt.xlabel('epochs')
plt.ylabel('Validation MAE')

# from the plot, epochs=80 optimizes the metric
model = build_model()
model.fit(train_data,train_targets,epochs=80,batch_size=32, verbose=0)
test_mse_score,test_mae_score = model.evaluate(test_data,test_targets)
print('the test MSE is {}'.format(test_mse_score))
print('the test MAE is {}'.format(test_mae_score))