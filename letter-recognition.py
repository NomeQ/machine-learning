# Letter Recognition using Perceptrons
# CS 545 Homework 1
# Naomi Dickerson 

import random
import csv
import numpy as np
import pickle
import os.path

# Learning rate
ADA = 0.2
MAX_EPOCH = 10

class Perceptron(object):
    """N dimensions and randomly initialized weights."""

    def __init__(self, dimensions):
        self.dimensions = dimensions
        # weights[0] will be the bias
        self.weights = ((np.random.rand(dimensions + 1)) * 2) - 1

    def get_weights(self):
        return (self.weights)

def import_data(my_file):
    """Return data formatted to appropriate numpy arrays."""
    letters = [[] for i in range(26)]
    with open(my_file) as csvfile:
        reader = csv.reader(csvfile)
        # Each instance will be indexed by letter and all 
        # attributes scaled between 0-1
        for row in reader:
            letters[s2idx(row[0])].append(to_vector(row[1::]))
    return letters

def divide_data(letters):
    """Return data split evenly into two sets."""
    training_data = []
    test_data = []
    for i in range(26):
        center = (len(letters[i]) // 2)
        training_data.append(letters[i][:center])
        test_data.append(letters[i][center::])
    return training_data, test_data

def train(data, perceptrons, i1, i2):
    """Train a single Perceptron i1 vs i2.

    arguments:
        data        -- A dataset with all training data for all classes
        perceptrons -- 2D array of all Perceptrons
        i1          -- Index of 1st letter, i.e. 0 == 'A'
        i2          -- Index of 2nd letter
    """
    # i1 should always be the smaller index
    if (i1 > i2):
        i1, i2 = i2, i1
    p = perceptrons[i1][(i2 - (i1 + 1))]
    w = p.weights
    # zip will truncate the longer data set, but that's fine here and 
    # helps ensure training is evenly balanced
    for i, j in zip(data[i1], data[i2]):
        # alternate between classes i1 and i2
        output = sgn(np.dot(w, i))
        if (output < 0):
            w = w + (ADA * i)
        output = sgn(np.dot(w, j))
        if (output > 0):
            w = w + (ADA * -1.0 * j)
    p.weights = w

def test(xs, perceptron):
    """Test a single instance and return the predicted class.

    arguments:
        xs         -- test instance, an array of 17 values
        perceptron -- 2D array of all Perceptrons
    """
    votes = [0] * 26
    for i in range(25):
        for j in range(i + 1, 26):
            w = perceptron[i][j - (i + 1)].weights
            output = sgn(np.dot(w, xs))
            if (output == 1):
                votes[i] += 1
            else:
                votes[j] += 1
    m = max(votes)
    # find all ties for max votes
    all_max = [i for i, j in enumerate(votes) if j == m]

    # Instead of random tie-breaking, run tied letters against
    # only one another and vote again. In spite of intuition,
    # this only slightly improves the accuracy of tie-breaks
    #l = len(all_max)
    #if l > 1:
        
        #new_votes = [0] * l
        #for i in range(l - 1):
        #    for j in range (i + 1, l):
        #        idx1 = all_max[i]
        #        idx2 = all_max[j]
        #        w = perceptron[idx1][idx2 - (idx1 + 1)].weights
        #        output = sgn(np.dot(w, xs))
        #        if (output == 1):
        #            new_votes[i] += 1
        #        else:
        #            new_votes[j] += 1
        #m1 = max(new_votes)
        #all_max1 = [i for i, j in enumerate(new_votes) if j == m1]
    #    return random.choice(all_max), True         
    return random.choice(all_max)

def run_tests(data, perceptron, c_matrix=None):
    """Return the accuracy of predictions on a dataset.

    arguments:
        data       -- dataset to test against
        perceptron -- a 2D array of all Perceptrons
        c_matrix   -- confusion matrix[26][26] (default = None)
    """
    correct = 0
    incorrect = 0
    for i in range(26):
        for j in data[i]:
            res = test(j, perceptron)
            if (res == i):
                correct += 1
                if c_matrix is not None:
                    c_matrix[i][i] += 1
            else:
                incorrect += 1
                if c_matrix is not None:
                    c_matrix[i][res] += 1
    
    return float(correct) / float(correct + incorrect)
     

def main():
    """Load dataset and train and test perceptrons using all-pairs."""
    # load sorted data if it exists
    if (os.path.isfile("sorted-data.pickle")):
        with open("sorted-data.pickle", "rb") as data:
            training_data, test_data = pickle.load(data)
    # otherwise, divide data in half into training and test sets
    else:
        all_letters = import_data("letter-recognition.data")
        training_data, test_data = divide_data(all_letters)
        with open("sorted-data.pickle", "wb") as data:
            pickle.dump((training_data, test_data), data)
    
    # create a 2-dimensional array of perceptrons for each letter pair combination
    perceptron = [[Perceptron(16) for y in range(25-x)] for x in range(26)]
    
    # train perceptrons until MAX_EPOCH is reached or accuracy ceases to improve
    old_acc = 0
    epoch = 0
    while (epoch < MAX_EPOCH):     
        
        # run all the training data against the current weights, initially
        # running against the set of randomly generated perceptrons
        new_acc = run_tests(training_data, perceptron)
        print "epoch {}: accuracy {:01.2f} on training set".format(epoch, new_acc)
        if (new_acc <= old_acc):
            break
        old_acc = new_acc
        epoch += 1

        # train all 325 perceptrons, where i, j are indices of the letter in
        # the alphabet
        for i in range(25):
            for j in range(i + 1, 26):
                train(training_data, perceptron, i, j)
    
    print "\n------- Running Perceptrons on Test Data --------"
    # Now run against the test data and generate a confusion matrix
    confusion = np.zeros((26, 26), dtype=np.int) 
    acc = run_tests(test_data, perceptron, confusion)
    print "Accuracy: {:01.2f}\n".format(acc)
    fancy_print(confusion)

### Some utility functions ###

def s2idx(letter):
    """Return index of character relative to 'A'."""
    return ord(letter) - ord("A")

def idx2s(num):
    """Return character from index"""
    return chr(num + ord('A'))

def to_vector(vals):
    """Create vector with scaled values from attribute list."""
    # also insert 1.0 at the beginning of each
    return np.insert((np.array(map(float, vals)) / 15), 0, 1.0) 

def sgn(x):
    """Return the sign of a value."""
    if x < 0:
        return -1
    else:
        return 1

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


