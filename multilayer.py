# Letter Recognition using a Neural Net
# CS 545 Homework 2
# Naomi Dickerson

# Ideas for better using numpy arrays in code, credit: 
# http://neuralnetworksanddeeplearning.com/chap1.html

import random
import csv
import numpy as np

# Learning rate
LEARN_R = 0.2
MOMENTUM = 0.1
MAX_EPOCH = 10
# Hidden units
H_UNIT = 64

class Neural_Net(object):

    def __init__(self, inputs, hidden, outputs):
        self.biases = [self.small_rands(s) for s in (hidden,outputs)]
        self.weights = [self.small_rands((h,s)) for h,s in [(hidden,inputs),(outputs,hidden)]]
        self.targets = [self.make_targets(t, outputs) for t in range(outputs)]

    def small_rands(self, size):
        """Return array randomized with -.25< w <.25 """
        return ((np.random.random_sample(size) - 0.5) / 2)

    def make_targets(self, target, outputs):
        """Return numpy array with target set to .9"""
        b = (np.ones(outputs)) * 0.1
        b[target] = 0.9
        return b

    def train(self, learning_rate, momentum, training_data, test_data, max_epochs):
        """Train neural net on a given data set, with given hyper-parameters"""
        for i in range(max_epochs):
            # Shuffle data for each epoch
            random.shuffle(training_data)
            delta_w_prev = [np.zeros(w.shape) for w in self.weights]
            delta_b_prev = [np.zeros(b.shape) for b in self.biases]
            for t,x in training_data:
                # Propogate forward, calculating activations
                # activations[0] are input units, activations[1], hidden units,etc
                activations = [x,[],[]]
                # Range is (layers - 1) of network, hardcoded here
                for n in range(2):
                    for b, w in zip(self.biases[n], self.weights[n]):
                        z = np.dot(w, activations[n]) + b
                        activations[n + 1].append(sigmoid(z))
                activations = map(np.array,activations)
                # retrive the appropriate target vector for calculating error
                target = self.targets[t]
                output_err = activations[2]*(1 - activations[2])*(target - activations[2])
                hidden_err = activations[1]*(1 - activations[1])*(np.dot(output_err, self.weights[1]))
                # This is kind of gross, programming-wise, but kept messing up matrix multiplication
                # update weights from hidden -> output units
                for j in range(26):
                    delta_w = learning_rate * output_err[j] * activations[1]
                    self.weights[1][j] += delta_w + (momentum * delta_w_prev[1][j])
                    delta_w_prev[1][j] = delta_w
                delta_b = learning_rate * output_err
                self.biases[1] += delta_b + (momentum * delta_b_prev[1])
                delta_b_prev[1] = delta_b
                # update weights from input -> hidden
                for j in range(H_UNIT):
                    delta_w = learning_rate * hidden_err[j] * activations[0]
                    self.weights[0][j] += delta_w + (momentum * delta_w_prev[0][j])
                    delta_w_prev[0][j] = delta_w
                delta_b = learning_rate * hidden_err
                self.biases[0] += delta_b + (momentum * delta_b_prev[0])
                delta_b_prev[0] = delta_b
            # At the end of the epoch, test accuracy on training data
            # as well as test data
            train_acc = self.test(training_data)
            test_acc = self.test(test_data)
            print "Epoch {}: Training Accuracy {:01.2f}\tTest Accuracy {:01.2f}".format(i,train_acc,test_acc)
      
    def classify(self, x):
        """Classify an instance with forward propagation, returns index of class"""
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x) + b)
        return np.argmax(x)

    def test(self, test_data):
        """Return the accuracy on a dataset"""
        total = len(test_data)
        correct = 0
        #conf_matrix = [[0] * 26 for x in range(26)]
        for t, x in test_data:
            guess = self.classify(x)
            #conf_matrix[t][guess] += 1
            if (guess == t):
                correct += 1
        acc = float(correct)/total
        return acc

def import_data(my_file):
    """Import a CSV file into tuples of format (t,x)"""
    letters = []
    with open(my_file) as csvfile:
        reader = csv.reader(csvfile)
        # Convert to tuple of form (0, [1.0,14.0..6.0])
        for row in reader:
            letters.append(makeInstance(row))
    return letters

def divide_data(letters):
    """Return data split evenly into two sets."""
    center = (len(letters) // 2)
    training_data = letters[:center]
    test_data = letters[center::]
    return training_data, test_data

def get_features(instances):
    """Return mean and std dev of all features in data set"""
    mean = [0 for x in range(16)]
    std_dev = [0 for x in range(16)]
    features = [[] for x in range(16)]
    for c, data in instances:
        for i in range(16):
            features[i].append(data[i])
    npfeatures = np.array(features)
    for i in range(0,16):
        mean[i] = np.mean(npfeatures[i])
        std_dev[i] = np.std(npfeatures[i])
    return mean, std_dev

def standardize(instances, mean, std_dev):
    """Standardize data set with arrays of mean, std_dev of features"""
    for c, data in instances:
        for i in range(16):
            data[i] = (data[i] - mean[i]) / std_dev[i]

def main():
    # Divide data in half into training and test sets
    all_letters = import_data("letter-recognition.data")
    training_data, test_data = divide_data(all_letters)
    
    # Get mean, std_dev from training data for all features
    mean, std_dev = get_features(test_data)
    
    # Now standardize all data with training data stats
    standardize(training_data, mean, std_dev)
    standardize(test_data, mean, std_dev)

    net = Neural_Net(16,H_UNIT,26)
    net.train(LEARN_R, MOMENTUM, training_data, test_data, MAX_EPOCH)
    
    
    # Now run against the test data and generate a confusion matrix
    #acc, matrix = net.test(test_data)
    #print "Test Accuracy: {:01.2f}\n".format(acc)
    #fancy_print(matrix)

### Some utility functions ###

def makeInstance(data):
    letter = s2idx(data[0])
    inputs = (map (float, data[1::]))
    return (letter, inputs)

def s2idx(letter):
    """Return index of character relative to 'A'."""
    return ord(letter) - ord("A")

def idx2s(num):
    """Return character from index"""
    return chr(num + ord('A'))

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def d_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def fancy_print(matrix):
    """Print a confusion matrix with labels."""
    alpha = map(chr, range(65, 91))
    print "   ",
    for letter in alpha:
        print "{:^3}".format(letter),
    print ""
    for i in range(26):
        print "{:^3}".format(alpha[i]),
        for num in matrix[i]:
            print "{:^3}".format(num),
        print ""

if __name__ == "__main__":
    main()


