# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:58:40 2020

Program reading order
1) mnist_explore.py
2) mnist_gradient_boost.py
    

GRADIENT BOOSTED DECISION TREES
-------------------------------
A more sophisticated model of gradient boosted decision trees is fitted to 
obtain a 96% accuracy rating on the test set. A large improvement over NB and
a small improvement over PCA+NB, but still 1/25 digits misclassified.


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
print(accuracy_score(ytest, y_pred_GB))     # 0.9666666666666667
                                            # 0.9733333333333334

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

plot_conf_mat(y_pred_GB)
plot_sample(y_pred_GB)


"""
A significant improvement over NB! However the difference in performance to 
PCA+NB is not huge considering the extra work required for this model - 
only a 2% increase in accuracy.

We need a more sophisticated model, e.g. a CNN that can use the inherent 
spatial information in the image.
"""


#%%
# Plot learning curve against dataset size
N, train_lc, test_lc = learning_curve(mod_GB, 
                                      Xtrain, ytrain, 
                                      train_sizes=np.linspace(0.3,1,10),
                                      cv=5)

plt.plot(N,np.mean(train_lc,1))
plt.plot(N,np.mean(test_lc,1))
plt.legend(["train", "test"])
plt.xlabel("training size")
plt.ylabel("accuracy")
plt.title("learning curve for gradient boost")

"""
This looks better than the NB! However this model might need more training 
data.
"""
