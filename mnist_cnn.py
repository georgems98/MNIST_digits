# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 19:35:11 2020

We continue exploring the MNIST digit set, trying to improve upon the efforts 
of our previous classifiers. 


CONVOLUTIONAL NEURAL NETWORK
----------------------------
Taking pity on my laptop, we use the smaller data set consisting of 8x8 images
provided by sklearn. Considering most people online have used the 28x28 set, 
this also gives us a chance to find our own architecture to fit this slightly
different data.

We start with the basic assumptions that a good structure will be 
    > some number of convolution layers
    > a dense layer between the convolution and final outputting dense layer
    > softmax for final activation

Pooling seems overkill with the 8x8 set when the first Conv2D layer reduces 
the dim of each slice to 6x6 already.


Ultimately we arrive at the following model with 99% test accuracy...

Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_5 (Conv2D)            (None, 6, 6, 64)          640       
_________________________________________________________________
flatten_5 (Flatten)          (None, 2304)              0         
_________________________________________________________________
dropout_9 (Dropout)          (None, 2304)              0         
_________________________________________________________________
dense_9 (Dense)              (None, 256)               590080    
_________________________________________________________________
dropout_10 (Dropout)         (None, 256)               0         
_________________________________________________________________
dense_10 (Dense)             (None, 10)                2570      
=================================================================
Total params: 593,290
Trainable params: 593,290
Non-trainable params: 0
_________________________________________________________________


@author: georgems
"""


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


sns.set()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix


#%% ==== LOAD DATA ====
# The train set size is 1347, the test set size is 450
# Images are 8x8        
from sklearn.datasets import load_digits
digits = load_digits()

X = digits.images
y = digits.target
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)


# ---- PLOT A SAMPLE
fig, axes = plt.subplots(10, 10, figsize=(8,8), 
                         subplot_kw={'xticks': [], 'yticks': []},
                         gridspec_kw = dict(hspace=0.1, wspace=0.1))

# Iterate over each ax in the axes array
for i, ax in enumerate(axes.flat):
    # Display the digit image in the given ax
    ax.imshow(Xtest[i], cmap='binary', interpolation='nearest')
    
    # Add the correct label in the corner
    ax.text(0.05, 0.05, str(ytest[i]), transform = ax.transAxes, 
            color = 'green')


#%% ==== CNN ====
# ---- DATA
# Expand dimensions to 4-D by adding axis in 4th dim
Xtrain = np.expand_dims(Xtrain, -1)
Xtest = np.expand_dims(Xtest, -1)

# Change precision to 32 bit and normalise pixel values.
Xtrain = Xtrain.astype("float32")/255
Xtest = Xtest.astype("float32")/255

# One hot encode the responses
from keras.utils import to_categorical

ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)


# ---- MODEL ARCHITECTURE
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


#%% How many layers?
def create_model(num_layers = 3):

    # Initialise
    model = Sequential()
    
    # Define the layers 
    model.add(Conv2D(16, 3, activation='relu', input_shape=(8,8,1)))
    
    if num_layers in [2,3]:
        model.add(Conv2D(32, 3, activation='relu'))
    
    if num_layers == 3:
        model.add(Conv2D(64, 3, activation='relu'))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    # Compile
    model.compile(optimizer="adam", loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

tune_mod = KerasClassifier(build_fn = create_model)

param_grid = dict(num_layers = [1,2,3])

grid = GridSearchCV(tune_mod, param_grid, cv=5)
grid.fit(Xtrain, ytrain, epochs = 20)

grid.cv_results_

"""
Similar results from all 3 models, so might as well use the simplest one for 
the sake of efficiency.

 'params': [{'num_layers': 1}, {'num_layers': 2}, {'num_layers': 3}],
 'split0_test_score': array([0.93703705, 0.92592591, 0.91481483]),
 'split1_test_score': array([0.93703705, 0.92962962, 0.92962962]),
 'split2_test_score': array([0.95539033, 0.95167285, 0.9628253 ]),
 'split3_test_score': array([0.94795537, 0.94423795, 0.91821563]),
 'split4_test_score': array([0.95910782, 0.95539033, 0.95539033]),
 'mean_test_score': array([0.94730552, 0.94137133, 0.93617514]),
 'std_test_score': array([0.009121  , 0.01172433, 0.01949926]),
 'rank_test_score': array([1, 2, 3])}

"""


#%% How many filters in each layer?
def create_model(num_filters = 32):

    # Initialise
    model = Sequential()
    
    # Define the layers 
    model.add(Conv2D(num_filters, 3, activation='relu', input_shape=(8,8,1)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    # Compile
    model.compile(optimizer="adam", loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

tune_mod = KerasClassifier(build_fn = create_model)

param_grid = dict(num_filters = [16,32,64,128])

grid = GridSearchCV(tune_mod, param_grid, cv=5)
grid.fit(Xtrain, ytrain, epochs = 20)

grid.cv_results_

"""
Best seems to be using 64 filters (although not by much).

 'params': [{'num_filters': 16}, {'num_filters': 32}, {'num_filters': 64}, {'num_filters': 128}],
 'split0_test_score': array([0.92962962, 0.94444442, 0.94814813, 0.94814813]),
 'split1_test_score': array([0.94444442, 0.95555556, 0.95185184, 0.95925927]),
 'split2_test_score': array([0.95539033, 0.94423795, 0.95539033, 0.95910782]),
 'split3_test_score': array([0.93680298, 0.94423795, 0.95910782, 0.95167285]),
 'split4_test_score': array([0.95539033, 0.96654278, 0.98141265, 0.9702602 ]),
 'mean_test_score': array([0.94433154, 0.95100373, 0.95918216, 0.95768965]),
 'std_test_score': array([0.01017283, 0.00890795, 0.01169668, 0.00761231]),
 'rank_test_score': array([4, 3, 1, 2])}

"""


#%% Size of the dense layer?
def create_model(density = 32):

    # Initialise
    model = Sequential()
    
    # Define the layers 
    model.add(Conv2D(64, 3, activation='relu', input_shape=(8,8,1)))
    model.add(Flatten())
    model.add(Dense(density, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    # Compile
    model.compile(optimizer="adam", loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

tune_mod = KerasClassifier(build_fn = create_model)

param_grid = dict(density = [16, 32, 64, 128, 256, 512])

grid = GridSearchCV(tune_mod, param_grid, cv=5)
grid.fit(Xtrain, ytrain, epochs = 20)

grid.cv_results_

"""
256 neurons in the dense layer is best - we get similar performance using 512
so might as well pick the smaller option

 'params': [{'density': 16}, {'density': 32}, {'density': 64}, {'density': 128}, {'density': 256}, {'density': 512}],
 'split0_test_score': array([0.89629632, 0.94814813, 0.95555556, 0.93333334, 0.94814813, 0.95925927]),
 'split1_test_score': array([0.92962962, 0.94074076, 0.95925927, 0.96296299, 0.96666664, 0.95555556]),
 'split2_test_score': array([0.93680298, 0.95910782, 0.9628253 , 0.96654278, 0.9628253 , 0.95539033]),
 'split3_test_score': array([0.9033457 , 0.93680298, 0.95167285, 0.95910782, 0.9628253 , 0.9702602 ]),
 'split4_test_score': array([0.93680298, 0.96654278, 0.97769517, 0.96654278, 0.98141265, 0.98141265]),
 'mean_test_score': array([0.92057552, 0.9502685 , 0.96140163, 0.95769794, 0.9643756 , 0.9643756 ]),
 'std_test_score': array([0.01729152, 0.01113175, 0.00895457, 0.01248843, 0.01061742, 0.01009549]),
 'rank_test_score': array([6, 5, 3, 4, 1, 1])}
"""


#%% Do we want dropout?
def create_model(drop1 = 0.5, drop2 = 0.5):

    # Initialise
    model = Sequential()
    
    # Define the layers 
    model.add(Conv2D(64, 3, activation='relu', input_shape=(8,8,1)))
    model.add(Flatten())
    model.add(Dropout(drop1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(drop2))
    model.add(Dense(10, activation='softmax'))
    
    # Compile
    model.compile(optimizer="adam", loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model


tune_mod = KerasClassifier(build_fn = create_model)


param_grid = dict(drop1=[0,0.2,0.4,0.5], drop2=[0,0.2,0.4,0.5])

grid = GridSearchCV(tune_mod, param_grid, cv=5)
grid.fit(Xtrain, ytrain, epochs = 20)

grid.cv_results_

"""
All combinations give fairly similar results with low standard deviation 
(~0.01), with the best mean test score coming from (0.2, 0.2).

[Full output is large thus ommitted]

"""


#%%
# ---- MODEL HYPERPARAMETERS
# Tune the batch size, epochs, learning rate using a randomised grid search

def create_model(lr=0.001):

    # Initialise
    model = Sequential()
    
    # Define the layers 
    model.add(Conv2D(64, 3, activation='relu', input_shape=(8,8,1)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    
    opt = Adam(learning_rate=lr)
    
    # Compile
    model.compile(optimizer=opt, loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model


tune_mod = KerasClassifier(build_fn = create_model)

batch_size = [1, 2, 4, 8, 16, 32, 64, 128, 256]
epochs = [20, 30, 50]
lr = [0.0001, 0.001]
param_grid = dict(batch_size=batch_size, epochs=epochs, lr=lr)

grid = RandomizedSearchCV(tune_mod, param_grid, cv=5, n_iter=10)
grid.fit(Xtrain[0:674,:,:,:], ytrain[0:674,:])

grid.cv_results_

"""
An initial sweep shows lr=0.1,0.01 are useless. A search using half the 
training data gives us (lr=0.001, epochs=50, batch_size=2) as good estimates.

    
 'params': [{'lr': 0.0001, 'epochs': 50, 'batch_size': 128},
  {'lr': 0.0001, 'epochs': 20, 'batch_size': 4},
  {'lr': 0.001, 'epochs': 20, 'batch_size': 16},
  {'lr': 0.001, 'epochs': 20, 'batch_size': 4},
  {'lr': 0.001, 'epochs': 50, 'batch_size': 2},
  {'lr': 0.001, 'epochs': 30, 'batch_size': 8},
  {'lr': 0.0001, 'epochs': 30, 'batch_size': 1},
  {'lr': 0.001, 'epochs': 50, 'batch_size': 128},
  {'lr': 0.001, 'epochs': 30, 'batch_size': 64},
  {'lr': 0.001, 'epochs': 50, 'batch_size': 64}],
 'split0_test_score': array([0.73333335, 0.8888889 , 0.96296299, 0.96296299, 0.95555556,
        0.92592591, 0.93333334, 0.93333334, 0.94074076, 0.94074076]),
 'split1_test_score': array([0.83703703, 0.91851854, 0.94074076, 0.92592591, 0.96296299,
        0.94074076, 0.93333334, 0.93333334, 0.93333334, 0.94074076]),
 'split2_test_score': array([0.85185188, 0.89629632, 0.95555556, 0.94814813, 0.96296299,
        0.94814813, 0.94814813, 0.94074076, 0.93333334, 0.94074076]),
 'split3_test_score': array([0.8296296 , 0.94074076, 0.94814813, 0.99259257, 0.96296299,
        0.95555556, 0.97037035, 0.94814813, 0.95555556, 0.96296299]),
 'split4_test_score': array([0.79104477, 0.93283582, 0.94776118, 0.95522386, 0.96268654,
        0.95522386, 0.93283582, 0.93283582, 0.91791046, 0.92537314]),
 'mean_test_score': array([0.80857933, 0.91545607, 0.95103372, 0.95697069, 0.96142621,
        0.94511884, 0.9436042 , 0.93767828, 0.93617469, 0.94211168]),
 'std_test_score': array([0.04264947, 0.02011796, 0.00758598, 0.02167715, 0.00293728,
        0.01102496, 0.01458776, 0.00600338, 0.01221635, 0.01200495]),
 'rank_test_score': array([10,  9,  3,  2,  1,  4,  5,  7,  8,  6])}   
    
    """


#%% ---- FIT MODEL

def create_model(lr=0.001):

    # Initialise
    model = Sequential()
    
    # Define the layers 
    model.add(Conv2D(64, 3, activation='relu', input_shape=(8,8,1)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    
    opt = Adam(learning_rate=lr)
    
    # Compile
    model.compile(optimizer=opt, loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model


mod_CNN = create_model(lr=0.001)
mod_CNN.fit(Xtrain, ytrain, 
            batch_size=2, epochs=50, 
            validation_data=(Xtest,ytest))

score = mod_CNN.evaluate(Xtest, ytest)
print("Test loss: ", score[0])          # Test loss:  0.07123944151523347
print("Test accuracy: ", score[1])      # Test accuracy:  0.9822221994400024


# Plot accuracy against epochs
plt.plot(mod_CNN.history.history["accuracy"])
plt.plot(mod_CNN.history.history["val_accuracy"])
plt.title("Accuracy vs epochs")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train", "test"])


"""
Accuracy plots suggest we can get away with 30-40 epochs here...

"""


#%% Is data augmentation going to be useful?
# Plot a learning curve of train/test accuracy against size of training data
tune_mod = KerasClassifier(build_fn=create_model, epochs=30)

N, train_lc, test_lc = learning_curve(tune_mod, 
                                      Xtrain, ytrain, 
                                      train_sizes=np.linspace(0.3,1,5),
                                      cv=3)

plt.plot(N,np.mean(train_lc,1))
plt.plot(N,np.mean(test_lc,1))
plt.legend(["train", "test"])


"""
The training and test curves don't appear to be close to convergence - we 
would probably benefit from additional data!

"""


#%% ---- FIT IMRPOVED MODEL 
# Image data generator with modest adjustments permitted - recall the image 
# sizes are only 8x8
datagen = ImageDataGenerator(rotation_range=5,
                             zoom_range=0.05,
                             shear_range=0.05,
                             width_shift_range=0.05,
                             height_shift_range=0.05)

def create_model(lr=0.001):

    # Initialise
    model = Sequential()
    
    # Define the layers 
    model.add(Conv2D(64, 3, activation='relu', input_shape=(8,8,1)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    
    opt = Adam(learning_rate=lr)
    
    # Compile
    model.compile(optimizer=opt, loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model


mod_CNN = create_model(lr=0.001)
mod_CNN.fit_generator(datagen.flow(Xtrain, ytrain, batch_size=2), 
                      epochs = 50, steps_per_epoch=Xtrain.shape[0]//2,
                      validation_data=(Xtest, ytest))
    
score = mod_CNN.evaluate(Xtest, ytest)
print("Test loss: ", score[0])          # Test loss:  0.01142376289425657
print("Test accuracy: ", score[1])      # Test accuracy:  0.995555579662323

# Plot accuracy against epochs
fig = plt.figure(figsize = [12,8], dpi=300)
plt.plot(mod_CNN.history.history["accuracy"])
plt.plot(mod_CNN.history.history["val_accuracy"])
plt.title("Accuracy vs epochs")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train", "test"])
plt.ylim(0.65,1)


"""
We hit over 99% test accuracy!
"""

mod_CNN.save(r"MNIST_digits/mod_CNN")


#%% ---- PREDICTION PLOTS
ytest2 = [p.argmax() for p in ytest]

def plot_conf_mat(y_pred):
    """
    Given an array of predictions on the test set, plot a confusion matrix of 
    the correct labels vs the predicted labels.
    """
    mat = confusion_matrix(ytest2, y_pred)

    sns.heatmap(mat, square = True, annot = True, cbar = False)
    plt.xlabel('predicted values')
    plt.ylabel('true values')
    

def plot_sample(y_pred):
    """
    Given an array of predictions on the test set, plot the first 100 
    images in the test set along with the predicted label.
    """
    # Initialise a figure of 10x10 axes
    fig, axes = plt.subplots(10, 10, figsize=(8,8), 
                         subplot_kw={'xticks': [], 'yticks': []},
                         gridspec_kw = dict(hspace=0.1, wspace=0.1))
    
    for i, ax in enumerate(axes.flat):
        ax.imshow((Xtest[i]*255).reshape(8,8), cmap='binary', interpolation='nearest')
        
        # Green label for correct label, red for incorrect
        ax.text(0.05, 0.05, str(y_pred[i]), transform=ax.transAxes, 
                color='green' if (ytest2[i] == y_pred[i]) else 'red')


y_pred = mod_CNN.predict(Xtest)
y_pred = [p.argmax() for p in y_pred]

plot_conf_mat(y_pred)
plot_sample(y_pred)

