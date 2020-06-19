# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:11:09 2020

Here we explore the MNIST digit set. The data set here consists of 8x8 images 
of digits, a sample of which can be displayed with matplotlib.


GAUSSIAN NAIVE BAYES
--------------------
A quick and easy Gaussian naive Bayes is fitted to obtain an 83% accuracy 
rating. This is pretty poor, but we would expect this 


GRADIENT BOOSTED DECISION TREES
-------------------------------
A more sophisticated model of gradient boosted decision trees is fitted to 
obtain a 95% accuracy rating on the test set.


@author: George Molina-Stubbs
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

# Iterate over each ax in the axes array
for i, ax in enumerate(axes.flat):
    # Display the digit image in the given ax
    ax.imshow(Xtest[i], cmap='binary', interpolation='nearest')
    
    # Add the correct label in the corner
    ax.text(0.05, 0.05, str(ytest[i]), transform = ax.transAxes, 
            color = 'green')
    

#%% ==== DIMENSION REDUCE ====
from sklearn.manifold import Isomap

# Initialise
iso = Isomap(n_components = 2)

# Fit to data and create the transformed data
iso.fit(digits.data)
data_projected = iso.transform(digits.data)

# Plot the 2D projected data with points coloured to match
plt.scatter(data_projected[:,0], data_projected[:,1], c=digits.target,
            edgecolor='none', alpha = 0.6, 
            cmap=plt.cm.get_cmap('Paired', 10))

# Add colour bar
plt.colorbar(label='digit label', ticks = range(10))
plt.clim(-0.5, 9.5)


#%% ==== NAIVE BAYES CLASSIFIER ====
# ---- MODEL
# Select model
from sklearn.naive_bayes import GaussianNB

# Initialise
mod_NB = GaussianNB()

# Fit to data
mod_NB.fit(Xtrain, ytrain)

# Predictions on test data
y_mod_NB = mod_NB.predict(Xtest)
accuracy_score(ytest, y_mod_NB) # 0.8333333333333334


# ---- PREDICTION PLOTS

def plot_conf_mat(y_pred):
    """
    Given an array of predictions on the test set, plot a confusion matrix of 
    the correct labels vs the predicted labels.
    """
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


plot_conf_mat(y_mod_NB)
plot_sample(y_mod_NB)
    
    
'''
Not a great classifier (83% accuray) but not bad for a first attempt with a 
quick fit with naive Bayes
'''



#%% ==== GRADIENT BOOSTING CLASSIFIER ====
# Change precision to 32 bit and normalise pixel values.
Xtrain = Xtrain.astype("float32")/255
Xtest = Xtest.astype("float32")/255

# ---- MODEL
from sklearn.ensemble import GradientBoostingClassifier


# ---- TUNE HYPERPARAMETERS
n_range = np.arange(1,10)
val_train, val_test = validation_curve(GradientBoostingClassifier(verbose=1),
                                       Xtrain, ytrain, 
                                       "n_estimators", n_estimators)

plt.plot(n_range, np.median(val_train, 1), color='blue', label='training score')
plt.plot(n_range, np.median(val_test, 1), color='red', label='validation score')
plt.legend(loc='best')





learning_rate = [0.005, 0.01, 0.1, 0.2]
n_estimators = np.arange(20,200,40)
max_depth = np.arange(1,10)
min_samples_split = np.arange(2,10)

# Try grid search over the boosting parameters
param_grid = dict(learning_rate = learning_rate,
                  n_estimators = n_estimators)

grid = RandomizedSearchCV(GradientBoostingClassifier(verbose=1), param_grid)
grid.fit(Xtrain, ytrain)

grid.cv_results_



#%%
# Fit to data
mod_GB = GradientBoostingClassifier(verbose=1)
mod_GB.fit(Xtrain, ytrain)

# Predictions on test data
y_mod_GB = mod_GB.predict(Xtest)
accuracy_score(ytest, y_mod_GB) # 0.9555555555555556



N, train_lc, test_lc = learning_curve(mod_GB, 
                                      Xtrain, ytrain, 
                                      train_sizes=np.linspace(0.3,1,5),
                                      cv=3)

plt.plot(N,np.mean(train_lc,1))
plt.plot(N,np.mean(test_lc,1))
plt.legend(["train", "test"])




# ---- PREDICTION PLOTS
plot_conf_mat(y_mod_GB)
plot_sample(y_mod_GB)


'''
A significant improvement over the Gaussian naive Bayes!
However 95% is still not great....
'''



#%% ---- Tune the model hyperparameters
# Compute test/train scores for different max_depth parameters
n_range = np.arange(1,10)
val_train, val_test = validation_curve(mod_GB, Xtest, ytest, 'max_depth', n_range )

plt.plot(n_range, np.median(val_train, 1), color='blue', label='training score')
plt.plot(n_range, np.median(val_test, 1), color='red', label='validation score')
plt.legend(loc='best')

''' 
The plot suggests max_depth = 2 or 3 gives best validation score
'''

# Try grid search over the boosting parameters
param_grid = {'n_estimators': np.array([50,75,100,125]),
           'learning_rate': np.array([0.05, 0.075, 0.1, 0.25, 0.5, 1])}

grid = GridSearchCV(GradientBoostingClassifier(), param_grid)

grid.fit(Xtrain, ytrain)
grid.best_params_ # {'learning_rate': 0.1, 'n_estimators': 125}


# Refit model with adjusted parameters
mod_GB2 = grid.best_estimator_.fit(Xtrain, ytrain)

y_mod_GB2 = mod_GB2.predict(Xtest)
accuracy_score(ytest, y_mod_GB2) # 0.9555555555555556

plot_conf_mat(y_mod_GB2)

'''
Not much improvement over the default parameters - should try a different 
method
'''


