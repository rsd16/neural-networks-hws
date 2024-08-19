# Project Payne, done by Alireza Rashidi Laleh
# use multi-class adaline network for optical character recognition.
# python 3.7.8


import numpy as np
import pandas as pd


class MultiClass_Adaline_Classifier:
    def __init__(self, alpha=0.0001, threshold = 0.001):
        self.bias = 1
        self.alpha = alpha
        self.threshold = threshold
        self.weights = {}
        self.b = {}
        self.x = {}
        self.y = 0
        self.net_input = {}
        self.output = {} # indices of this list ==> 0: X and 1: O ... and fyi, label values ==> X =: 1 and O =: -1

        self.maximum_delta_weights = {}
        self.condition = False

    def fit(self, x_train, y_train):
        x_train = x_train
        y_train = y_train
        while self.condition == False:
            for k in range(0, len(x_train)):
                record = x_train[k]
                for label in np.unique(y_train): # {1: all the stuff about label X ... -1: all the stuff about label O}
                    if k == 0:
                        self.b[label] = 0
                        temporary = [0] * len(record)
                        self.weights[label] = temporary
                        for j in range(0, len(record)):
                            self.maximum_delta_weights[j] = 0

                    for j in range(0, len(record)):
                        self.x[j] = record[j]

                    weighted_sum = 0
                    for j in range(0, len(record)):
                        weighted_sum += self.weights[label][j] * self.x[j]

                    self.net_input[label] = self.b[label] + weighted_sum

                    delta_weights = {}
                    for j in range(0, len(record)):
                        delta_weights[j] = self.alpha * self.x[j] * (y_train[k] - self.net_input[label])
                        if abs(delta_weights[j]) > abs(self.maximum_delta_weights[j]):
                            self.maximum_delta_weights[j] = abs(delta_weights[j])

                    temporary = []
                    next_weights = 0
                    for j in range(0, len(record)):
                        next_weights = self.weights[label][j] + delta_weights[j]
                        temporary.append(next_weights)

                    self.weights[label] = temporary

                    delta_b = self.alpha * self.bias * (y_train[k] - self.net_input[label])
                    self.b[label] += delta_b

            max_difference = max(self.maximum_delta_weights.values())
            if abs(max_difference) < self.threshold:
                print('Breaking out the algorithm...')
                self.condition = True
                break
            else:
                self.maximum_delta_weights = {key: 0 for key in self.maximum_delta_weights}
                

    def predict(self, x_test, y_test):
        x_test = x_test
        y_test = y_test
        results = []
        for k in range(0, len(x_test)):
            record = x_test[k]
            for label in np.unique(y_test):
                for j in range(0, len(record)):
                    self.x[j] = record[j]

                weighted_sum = 0
                for j in range(0, len(record)):
                    weighted_sum += self.weights[label][j] * self.x[j]

                self.net_input[label] = self.b[label] + weighted_sum

                if self.net_input[label] >= 0:
                    self.output[label] = 1
                else:
                    self.output[label] = -1

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


records = np.loadtxt('preprocessed_data.txt')
np.random.shuffle(records)

X = records[:, 0:-1]
labels = records[:, -1]

split_size = int(len(X) * 2 / 3.0)

x_train = X[:split_size]
y_train = labels[:split_size]
x_test = X[split_size:]
y_test = labels[split_size:]

#print(len(x_train))
#print(len(y_train))
#print(len(x_test))
#print(len(y_test))

#print(x_train)
#print('####################################################')
#print(x_test)

model = MultiClass_Adaline_Classifier()
#print(model)

model.fit(x_train, y_train)
predictions = model.predict(x_test, y_test)

print(y_test)
print(predictions)

accuracy = model.score(y_test, predictions)
print(accuracy)
