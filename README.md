# machine-learning
Code created for PSU's machine learning class. Based around the [UCI Letter Recognition](http://archive.ics.uci.edu/ml/datasets/Letter+Recognition) 
dataset, it is not general purpose.

# letter-recognition.py
Letter recognition using simple binary perceptrons. Uses all-pairs method to classify instances and produces 
a confusion matrix of the results. Achieves aroung 70% accuracy on test set.

# multilayer.py
A multilayer neural net with 1 hidden layer. Performs best with eta=.3, momentum=.3, and at least 64 hidden units,
achieving over 80% accuracy.

