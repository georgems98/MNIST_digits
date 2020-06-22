# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:11:09 2020

Here we explore the MNIST digit set. The data set here consists of 8x8 images 
of digits, a sample of which can be displayed with matplotlib.


GAUSSIAN NAIVE BAYES
--------------------
A quick and easy Gaussian naive Bayes is fitted to obtain an 83% accuracy 
rating. This is pretty poor, but we would expect this as pixel values won't 
be independent.


GRADIENT BOOSTED DECISION TREES
-------------------------------
A more sophisticated model of gradient boosted decision trees is fitted to 
obtain a 96% accuracy rating on the test set. A large improvement over NB, but
still 1/25 digits misclassified.


@author: georgems
"""


import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline


#%% ==== LOAD DATA ====
# The train set size is 1347, the test set size is 450
# Images are 8x8        
from sklearn.datasets import load_digits
digits = load_digits()

X = digits.data
y = digits.target
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)


# ---- PLOT A SAMPLE
fig, axes = plt.subplots(10, 10, figsize=(8,8), 
                         subplot_kw={'xticks': [], 'yticks': []},
                         gridspec_kw = dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    # Display the digit image in the given ax
    ax.imshow(Xtest[i].reshape(8,8), cmap='binary', interpolation='nearest')
    
    # Add the correct label in the corner
    ax.text(0.05, 0.05, str(ytest[i]), transform = ax.transAxes, 
            color = 'green')
    

#%% ==== DIMENSION REDUCE ====
# Project the 64 dim predictor space into 2D to get a feel for the data
from sklearn.manifold import Isomap

iso = Isomap(n_components = 2)

# Fit to data and create the transformed data
iso.fit(digits.data)
data_projected = iso.transform(digits.data)

# Plot the 2D projected data with points coloured to match classifications
plt.scatter(data_projected[:,0], data_projected[:,1], c=digits.target,
            edgecolor='none', alpha = 0.6, 
            cmap=plt.cm.get_cmap('Paired', 10))

plt.colorbar(label='digit label', ticks = range(10))
plt.clim(-0.5, 9.5)


#%% ==== NAIVE BAYES CLASSIFIER ====
# ---- MODEL
from sklearn.naive_bayes import GaussianNB

mod_NB = GaussianNB()
mod_NB.fit(Xtrain, ytrain)

# Predictions on test data
y_pred_NB = mod_NB.predict(Xtest)
accuracy_score(ytest, y_pred_NB) # 0.8333333333333334


# ---- PREDICTION PLOTS

def plot_conf_mat(y_pred):
    """
    Given an array of predictions on the test set, plot a confusion matrix of 
    the correct labels vs the predicted labels.
    """
    plt.figure()
    mat = confusion_matrix(ytest, y_pred)

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
        ax.imshow(Xtest[i].reshape(8,8), cmap='binary', interpolation='nearest')
        
        # Green label for correct label, red for incorrect
        ax.text(0.05, 0.05, str(y_pred[i]), transform=ax.transAxes, 
                color='green' if (ytest[i] == y_pred[i]) else 'red')


plot_conf_mat(y_pred_NB)
plot_sample(y_pred_NB)
    
"""
Not a great classifier (83% accuray) but not bad for an easily fitted first 
attempt. Considering we're working with image data we wouldn't expect the 
pixel values to be independent anyway.
"""


#%% ==== NB WITH PCA ====
# Change precision to 32 bit and normalise pixel values.
Xtrain = Xtrain.astype("float32")/255
Xtest = Xtest.astype("float32")/255


# ---- MODEL
from sklearn.decomposition import PCA

# 
pca = PCA().fit(Xtrain)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("n_components")
plt.ylabel("cumulative explained variance ratio")

pca = PCA()
nb = GaussianNB()

mod_NB2 = make_pipeline(pca, nb)

grid_params = dict(pca__n_components=np.arange(2,65))
grid = GridSearchCV(mod_NB2, grid_params, cv=10).fit(Xtrain, ytrain)

"""
{'pca__n_components': 41}
"""

mod_NB2 = grid.best_estimator_.fit(Xtrain, ytrain)




Xtest_trans = pca.inverse_transform(pca.transform(Xtest))

fig, axes = plt.subplots(10, 10, figsize=(8,8), 
                         subplot_kw={'xticks': [], 'yticks': []},
                         gridspec_kw = dict(hspace=0.1, wspace=0.1))

# Iterate over each ax in the axes array
for i, ax in enumerate(axes.flat):
    # Display the digit image in the given ax
    ax.imshow(Xtest_trans[i].reshape(8,8), cmap='binary', interpolation='nearest')
    
    # Add the correct label in the corner
    ax.text(0.05, 0.05, str(ytest[i]), transform = ax.transAxes, 
            color = 'green')










mod_NB2.fit(Xtrain, ytrain)

# ---- PREDICTIONS
y_pred_NB2 = mod_NB2.predict(Xtest)
print(accuracy_score(ytest, y_pred_NB2))      # 0.94

plot_conf_mat(y_pred_NB2)
plot_sample(y_pred_NB2)

"""
Looks like we need to find a more sophisticated classifier, with 94% accuracy 
as our baseline success.
"""


#%% ==== GRADIENT BOOSTING CLASSIFIER ====
# Change precision to 32 bit and normalise pixel values.
Xtrain = Xtrain.astype("float32")/255
Xtest = Xtest.astype("float32")/255

# ---- MODEL
from sklearn.ensemble import GradientBoostingClassifier


#---- TUNE HYPERPARAMETERS
#%% How many trees?

def val_plot(mod, param, param_range):
    val_train, val_test = validation_curve(mod, Xtrain, ytrain, 
                                           param, param_range,
                                           cv=3)
    
    plt.figure()
    plt.plot(param_range, np.median(val_train, 1), color='blue', label='training score')
    plt.plot(param_range, np.median(val_test, 1), color='red', label='validation score')
    plt.legend(loc='best')
    plt.xlabel(param)
    plt.ylabel("accuracy")
    plt.title("Accuracy vs " + param)
    
    


n_estimators = np.arange(40,200,20)
mod = GradientBoostingClassifier(verbose=1)
val_plot(mod, "n_estimators", n_estimators)

"""
Somewhere in the range of 80 - 120 seems good, with peak val score at 110
"""


#%% Randomised search across a range of parameters
learning_rate = [0.005, 0.01, 0.1, 0.2]
n_estimators = np.arange(100,120,5)
max_depth = np.arange(1,20)
min_samples_split = np.arange(2,20)

param_grid = dict(learning_rate = learning_rate,
                  n_estimators = n_estimators,
                  max_depth = max_depth,
                  min_samples_split = min_samples_split)

grid = RandomizedSearchCV(GradientBoostingClassifier(verbose=1), param_grid)
grid.fit(Xtrain, ytrain)

grid.cv_results_

"""
The top three combinations are
  {'n_estimators': 110, 'min_samples_split': 8, 'max_depth': 5, 'learning_rate': 0.1},
  {'n_estimators': 100, 'min_samples_split': 10, 'max_depth': 10, 'learning_rate': 0.2},
  {'n_estimators': 110, 'min_samples_split': 7, 'max_depth': 12, 'learning_rate': 0.1}
"""


#%% Max depth and minimum samples to split on?
mod = GradientBoostingClassifier(n_estimators=110, learning_rate = 0.1,
                                 verbose=1)

max_depth = np.arange(4,13,2)
val_plot(mod, "max_depth", max_depth)

min_samples_split = np.arange(2,11,2)
val_plot(mod, "min_samples_split", min_samples_split)

"""
Looks like max_depth in range 3-7 and min_samples_split less than 5 is good.
We take the lower end of these for simplicity.
"""


#%% Size of subsampling?
param_grid = dict(subsample = np.arange(0.6,1,0.05))

grid = GridSearchCV(GradientBoostingClassifier(verbose=1, 
                                                     learning_rate=0.1, 
                                                     n_estimators = 110,
                                                     max_depth=3,
                                                     min_samples_split=2), 
                          param_grid, cv = 3)
grid.fit(Xtrain, ytrain)

grid.cv_results_ 

"""
{'subsample': 0.65}
"""


#%% ---- FIT MODEL
mod_GB = GradientBoostingClassifier(learning_rate=0.1, n_estimators=110,
                                    max_depth=3, min_samples_split=2,
                                    subsample=0.65,
                                    verbose=1)
mod_GB.fit(Xtrain, ytrain)

y_pred_GB = mod_GB.predict(Xtest)
print(accuracy_score(ytest, y_pred_GB))     # 0.9644444444444444


#%%
# Plot learning curve against dataset size
N, train_lc, test_lc = learning_curve(mod_GB, 
                                      Xtrain, ytrain, 
                                      train_sizes=np.linspace(0.3,1,5),
                                      cv=3)

plt.plot(N,np.mean(train_lc,1))
plt.plot(N,np.mean(test_lc,1))
plt.legend(["train", "test"])

"""
Looks like we haven't converged with this training data
"""


#%% ---- PREDICTION PLOTS
plot_conf_mat(y_pred_GB)
plot_sample(y_pred_GB)

'''
A significant improvement over the Gaussian naive Bayes!
However 96% is still not great...
We need a more sophisticated model, e.g. a CNN that can use the inherent 
spatial information in the image.
'''

