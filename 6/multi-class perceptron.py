# Project Sclerosis, done by Alireza Rashidi Laleh
# use multi-class perceptron for optical character recognition.
# python 3.7.8


import numpy as np
import pandas as pd


class MultiClass_Perceptron_Classifier:
    def __init__(self, alpha=0.001, theta=0.00001):
        self.bias = 1
        self.alpha = alpha
        self.theta = theta
        self.weights = {}
        self.b = {}
        self.x = []
        self.y = 0
        self.net_input = {}
        self.output = {} # indices of this list ==> 0: X and 1: O ... and fyi, label values ==> X =: 1 and O =: -1

    def fit(self, x_train, y_train):
        x_train = x_train
        y_train = y_train
        for k in range(0, len(x_train)):
            record = x_train[k]
            for label in np.unique(y_train): # {1: all the stuff about label X ... -1: all the stuff about label O}
                if k == 0:
                    self.b[label] = 0
                    temporary = [0] * len(record)
                    self.weights[label] = temporary

                x = {}
                for j in range(0, len(record)):
                    x[j] = record[j]

                weighted_sum = 0
                for j in range(0, len(record)):
                    weighted_sum += self.weights[label][j] * x[j]

                self.net_input[label] = self.b[label] + weighted_sum

                if self.net_input[label] > self.theta:
                    exit_neuron = 1
                elif self.net_input[label] <= self.theta and self.net_input[label] >= -self.theta: # how to get rid of this?
                    exit_neuron = 0
                elif self.net_input[label] < -self.theta:
                    exit_neuron = -1

##                if label == 0:
##                    if exit_neuron != 0:
##                        exit_neuron = 0
##
##                if label == -1:
##                    if exit_neuron != -1:
##                        exit_neuron = 0

                self.output[label] = exit_neuron

                if self.output[label] != y_train[k]:
                    delta_weights = {}
                    for j in range(0, len(record)):
                        delta_weights[j] = self.alpha * x[j] * y_train[k]

                    temporary = []
                    next_weights = 0
                    for j in range(0, len(record)):
                        next_weights = self.weights[label][j] + delta_weights[j]
                        temporary.append(next_weights)

                    self.weights[label] = temporary

                    delta_b = self.alpha * self.bias * y_train[k]
                    self.b[label] += delta_b

            if self.output[1] == self.output[-1]:
                key = max(self.net_input, key=self.net_input.get)
                self.output[key] == 1
                other_label = list(self.output.keys())
                other_label.remove(key)
                for item in other_label:
                    self.output[item] = 0


    def predict(self, x_test, y_test):
        x_test = x_test
        y_test = y_test
        results = []
        for k in range(0, len(x_test)):
            record = x_test[k]
            for label in np.unique(y_train):
                x = {}
                for j in range(0, len(record)):
                    x[j] = record[j]

                weighted_sum = 0
                for j in range(0, len(record)):
                    weighted_sum += self.weights[label][j] * x[j]

                self.net_input[label] = self.b[label] + weighted_sum

                if self.net_input[label] > self.theta:
                    exit_neuron = 1
                elif self.net_input[label] <= self.theta and self.net_input[label] >= -self.theta: # how to get rid of this?
                    exit_neuron = 0
                elif self.net_input[label] < -self.theta:
                    exit_neuron = -1

                if label == 1:
                    if exit_neuron != 1:
                        exit_neuron = 0

                if label == 0:
                    if exit_neuron != -1:
                        exit_neuron = 0

                self.output[label] = exit_neuron

            if self.output[1] == self.output[-1]:
                key = max(self.net_input, key=self.net_input.get)
                self.output[key] == 1
                other_label = list(self.output.keys())
                other_label.remove(key)
                for item in other_label:
                    self.output[item] = 0

            key = max(self.output, key=self.output.get)
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


X = pd.read_csv('input.txt', delimiter='\t', header=None)

labels = pd.read_csv('targets.txt', delimiter='\t', header=None)
labels['class'] = labels.idxmax(1)
cols = [col for col in labels.columns if col in ['class']]
labels = labels[cols]

df = X.join(labels)
df = df.sample(frac=1).reset_index(drop=True)

labels2 = df['class']
X2 = df.drop(['class'], axis=1)

X = np.array(X2)
labels = np.array(labels2)
labels = labels.ravel()

split_size = int(len(X) * 2 / 3.0)

x_train = X[:split_size]
y_train = labels[:split_size]
x_test = X[split_size:]
y_test = labels[split_size:]

model = MultiClass_Perceptron_Classifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test, y_test)

print('Actual targets:')
print(y_test)
print('Predicted Targets:')
print(predictions)

accuracy = model.score(y_test, predictions)
print(accuracy)
