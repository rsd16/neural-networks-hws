# Project Past, done by Alireza Rashidi Laleh
# use multi-class perceptron for optical character recognition.
# python 3.7.8


import numpy as np
import pandas as pd
import scipy.io
import itertools
from sklearn.neural_network import MLPClassifier


class MultiLayer_Perceptron_Classifier:
    def __init__(self, alpha=0.001, theta=0.0001):
        self.bias = 1
        self.alpha = alpha
        self.theta = theta
        self.x = {}
        self.z = {}
        self.b_z = {}
        self.weights_z = {}
        self.net_input_z = {}
        self.net_input_y = {}
        self.b_y = {}
        self.weights_y = {}
        self.output_y = {}


    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x))) # binary sigmoid
        #return (2 / (1 + np.exp(-x))) - 1 # bipolar sigmoid


    def sigmoid_derivative(self, f):
        return f * (1 - f) # binary sigmoid derivative
        #return 0.5 * (1 + f) * (1 - f) # bipolar sigmoid derivative


    def instantiate_weights(self, length, labels):
        length = length
        labels = labels
        temporary = []
        for i in range(0, length):
            for j in range(0, length):
                temporary.append(np.random.uniform(-0.5, 0.5))

            self.weights_z[i] = temporary
            self.b_z[i] = np.random.uniform(-0.5, 0.5)

        temporary = []
        for label in labels:
            temporary = []
            for j in range(0, length):
                temporary.append(np.random.uniform(-0.5, 0.5))

            self.weights_y[label] = temporary
            self.b_y[label] = np.random.uniform(-0.5, 0.5)


    def feedforward(self, x_data, y_data):
        record = x_data
        label = y_data

        # for input layer:
        for j in range(0, len(record)):
            self.x[j] = record[j]

        # for hidden layer:
        for j in range(0, len(record)):
            temporary = self.weights_z[j] # temporary now is a list
            weighted_sum = 0
            for i in range(0, len(record)):
                weighted_sum += temporary[i] * self.x[i]

            self.z[j] = self.b_z[j] + weighted_sum
            self.net_input_z[j] = self.sigmoid(self.z[j])

        # for output layer:
        for j in range(0, len(record)):
            temporary = self.weights_y[label]
            weighted_sum = 0
            for i in range(0, len(record)):
                weighted_sum += temporary[i] * self.net_input_z[i]

            self.net_input_y[label] = self.b_y[label] + weighted_sum
            self.output_y[label] = self.sigmoid(self.net_input_y[label])


    def backpropagation(self, x_train, y_train):
        record = x_train
        label = y_train

        delta_factor_y = (label - self.output_y[label]) * self.sigmoid_derivative(self.output_y[label])

        delta_weights_zy = {}
        for j in range(0, len(record)):
            delta_weights_zy[j] = self.alpha * self.z[j] * delta_factor_y

        delta_b_y = self.alpha * delta_factor_y

        weighted_sum = 0
        for j in range(0, len(record)):
            weighted_sum += delta_factor_y * self.weights_y[label][j]

        delta_factor_z = {}
        for j in range(0, len(record)):
            delta_factor_z[j] = weighted_sum * self.sigmoid_derivative(self.net_input_z[j])

        delta_weights_xz = {}
        for j in range(0, len(record)):
            delta_weights_xz[j] = self.alpha * self.x[j] * delta_factor_z[j]

        delta_b_z = {}
        for j in range(0, len(record)):
            delta_b_z[j] = self.alpha * delta_factor_z[j]

        temporary = []
        next_weights_y = 0
        for j in range(0, len(record)):
            next_weights_y = self.weights_y[label][j] + delta_weights_zy[j]
            temporary.append(next_weights_y)

        self.weights_y[label] = temporary

        for j in range(0, len(record)):
            self.weights_z[j] = self.weights_z[j] + delta_weights_xz[j]

        self.b_y[label] += delta_b_y

        for j in range(0, len(record)):
            self.b_z[j] += delta_b_z[j]

    def fit(self, x_train, y_train, epochs=1):
        x_train = x_train
        y_train = y_train
        epochs = epochs

        for i in range(0, epochs):
            for j in range(0, len(x_train)):
                if j == 0:
                    self.instantiate_weights(len(x_train[j]), np.unique(y_train))

                for label in np.unique(y_train):
                    self.feedforward(x_train[j], y_train[j])
                    self.backpropagation(x_train[j], y_train[j])

        #print(self.output_y)



##                 if self.output_y[0] == self.output_y[1] or self.output_y[0] == self.output_y[2] or self.output_y[1] == self.output_y[2] or self.output_y[0] == self.output_y[1] == self.output_y[2]:
##                     key = max(self.output_y, key=self.output_y.get)
##                     self.output_y[key] == 1
##                     other_labels = list(self.output_y.keys())
##                     other_labels.remove(key)
##                     for item in other_label:
##                         self.output_y[item] = 0

            # for batch-updating. if we want to do this, we have to change the called method too.
            #self.backpropagation(x_train[j], y_train[j])

    def predict(self, x_test, y_test):
        x_test = x_test
        y_test = y_test
        results = []

        for j in range(0, len(x_test)):
            self.feedforward(x_test[j], y_test[j])
            key = max(self.output_y, key=self.output_y.get)
            results.append(key)

        return results

    def score(self, y_test, predictions):
        predictions = predictions
        y_test = y_test
        true_counts = 0
        false_counts = 0
        for j in range(0, len(y_test)):
            if y_test[j] == predictions[j]:
                true_counts += 1
            else:
                false_counts += 1

        print(f'True Counts: {true_counts}')
        print(f'False Counts: {false_counts}')
        accuracy = float(true_counts) / float(true_counts + false_counts)
        return accuracy


###########################################################################################################################
# my method
###########################################################################################################################

print('My humble method:')

X = pd.read_csv('input.txt', delimiter='\t', header=None)

labels = pd.read_csv('targets.txt', delimiter='\t', header=None)

labels['class'] = labels.idxmax(1)
##print(labels['class'].value_counts())
##print(labels.tail())

cols = [col for col in labels.columns if col in ['class']]
labels = labels[cols]

df = X.join(labels)

#print(df.head())

df = df.sample(frac=1).reset_index(drop=True)
##print(df.head())
##print(df.tail())
##print(df.columns)

labels2 = df['class']
X2 = df.drop(['class'], axis=1)

##print(X.head())
##print(labels.head())

X = np.array(X2)
#print(X)

labels = np.array(labels2)
#print(labels)
labels = labels.ravel()
#print(labels)

split_size = int(len(X) * 2 / 3.0)

x_train = X[:split_size]
y_train = labels[:split_size]
x_test = X[split_size:]
y_test = labels[split_size:]

##print(len(x_train))
##print(len(y_train))
##print(len(x_test))
##print(len(y_test))

model = MultiLayer_Perceptron_Classifier()
#print(model)

model.fit(x_train, y_train)
predictions1 = model.predict(x_test, y_test)

#print(y_test)
#print(predictions1)

accuracy1 = model.score(y_test, predictions1)
print(accuracy1)
##
#############################################################################################################################
### scikit-learn's method
#############################################################################################################################
##
print('Scikitlearn method:')
clf = MLPClassifier()

clf.fit(x_train, y_train)
predictions2 = clf.predict(x_test)

#accuracy2 = model2.score(predictions2, y_test)
accuracy2 = model.score(y_test, predictions2)

print('Scikit-Learn method:')
print(accuracy2)
