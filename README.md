# MNIST_digits
An exploration of the canonical example for computer vision, the MNIST handwritten digit set. A couple of statistical learning algorithms are fitted to the data first to get a sense of what works and what doesnt. A CNN is fitted afterwards, and achieves 99.5% accuracy on the training data. Note the specific version of the data set used; sklearn provides a smaller set of 8x8 images whereas Keras provides a large set of 28x28 images.

### Program listings
* mnist_explore.py is the initial exploration of the data set, and fits two Gaussian NB models to obtain a baseline accuracy.
* mnist_gradient_boost.py fits gradient boosted decision trees in an attempt to improve upon NB
* mnist_cnn.py is the CNN fitted using Keras
* mod_CNN is the final fitted Keras model
* package_list.txt is a list of packages installed in the working environment
