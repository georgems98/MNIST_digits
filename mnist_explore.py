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


PRINCIPAL COMPONENT ANALYSIS + NB
---------------------------------
Reducing the input dimensions to 41-dim with PCA offers a big improvement over
NB at a low computational cost, giving us 94% accuracy as a baseline for more
complex models.


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

    
# ---- Learning curve plot
N, train_lc, test_lc = learning_curve(GaussianNB(), 
                                      Xtrain, ytrain, 
                                      train_sizes=np.linspace(0.3,1,20),
                                      cv=50)
    
plt.plot(N, np.mean(train_lc,1), label="train")
plt.plot(N, np.mean(test_lc,1), label="test")
plt.legend(loc="best")
plt.xlabel("training size")
plt.ylabel("accuracy")
plt.title("learning curve for NB")


"""
Not a great classifier at 83% accuray but not entirely terrible for an easily 
fitted first attempt. 

Plotting the learning curve shows the test score increasing up to a point and
then decreasing - possible overfitting to the training data.

Let's reduce the dimensions of the input data...
"""



#%% ==== NB WITH PCA ====
# Change precision to 32 bit and normalise pixel values.
Xtrain = Xtrain.astype("float32")/255
Xtest = Xtest.astype("float32")/255


# ---- MODEL
from sklearn.decomposition import PCA


# ---- TUNE N_COMPONENTS
# Quick plot of the explained variance ratio against n_components
pca = PCA().fit(Xtrain)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("n_components")
plt.ylabel("cumulative explained variance ratio")

# Set pipepline and perform a grid search - very quick so search all values
pca = PCA()
nb = GaussianNB()
mod_NB2 = make_pipeline(pca, nb)

grid_params = dict(pca__n_components=np.arange(2,65))
grid = GridSearchCV(mod_NB2, grid_params, cv=10).fit(Xtrain, ytrain)

"""
{'pca__n_components': 41}
"""

#%% We can plot the transformed data 
pca = PCA(41).fit(Xtrain)
Xtest_trans = pca.inverse_transform(pca.transform(Xtest))

fig, axes = plt.subplots(10, 10, figsize=(8,8), 
                         subplot_kw={'xticks': [], 'yticks': []},
                         gridspec_kw = dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(Xtest_trans[i].reshape(8,8), cmap='binary', interpolation='nearest')
    
    ax.text(0.05, 0.05, str(ytest[i]), transform = ax.transAxes, 
            color = 'green')

"""
We can verify visually that with 41 components the image data is completely 
recognisable. Some additional noise seems to have been introduced, which might
help with the overfitting. 
"""


#%% ---- FIT
mod_NB2 = grid.best_estimator_.fit(Xtrain, ytrain)
mod_NB2.fit(Xtrain, ytrain)


# ---- PREDICTIONS
y_pred_NB2 = mod_NB2.predict(Xtest)

print(accuracy_score(ytest, y_pred_NB2))      # 0.9422222222222222

plot_conf_mat(y_pred_NB2)
plot_sample(y_pred_NB2)

"""
A pretty big improvement with 94.2% accuracy - but not perfect.
Looks like we need to find a more sophisticated classifier, with 94% accuracy 
as our baseline success.
"""
