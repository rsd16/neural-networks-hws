# Project Despair, done by Alireza Rashidi Laleh
# AND logic gate done using Adaline network.
# python 3.7.8


import numpy as np
import matplotlib.pyplot as plt


X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
labels = np.array([1, -1, -1, -1])
labels_colors = [-1, -1, -1, 1]

dummy_x1 = np.array([-2, 2])
dummy_x2 = np.array([0, 0])

xx1 = [-1, -1, 1, 1]
xx2 = [-1, 1, -1, 1]


def show_plot(j, next_b, next_weights):
    for i in range(0, 2):
        dummy_x2[i] = (-next_b - (next_weights[0] * dummy_x1[i])) / next_weights[1]

    fig = plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    plt.scatter(xx1, xx2, c=labels_colors, label=labels, linewidth=6)
    plt.plot(dummy_x1, dummy_x2, linewidth=5, color='r')
    #plt.plot(dummy_x1, dummy_x22, linewidth=5, color='b')
    plt.title(f'Plot after reading record {j}')
    plt.show()


class Adaline:
    def __init__(self, alpha=0.0001, threshold=0.001):
        self.bias = 1
        self.alpha = alpha
        self.threshold = threshold
        self.x = {}
        self.weights = {}
        self.b = []
        self.net_input = 0
        self.maximum_delta_weights = {}
        self.condition = False
        self.output = 0
        

    def fit(self, x_train, y_train):
        x_train = x_train
        y_train = y_train
        while self.condition == False:
            for k in range(0, len(x_train)):
                record = x_train[k]
                if k == 0:
                    self.b = [0] * len(record)
                    for j in range(0, len(record)):
                        self.weights[j] = 0
                        self.maximum_delta_weights[j] = 0
                        self.b[j] = 0

                for j in range(0, len(record)):
                    self.x[j] = record[j]

                weighted_sum = 0
                for j in range(0, len(record)):
                    weighted_sum += self.weights[j] * self.x[j]

                self.net_input = self.b + weighted_sum

                delta_weights = {}
                for j in range(0, len(record)):
                    delta_weights[j] = self.alpha * self.x[j] * (y_train[k] - self.net_input)
                    if abs(delta_weights[j]) > abs(self.maximum_delta_weights[j]):
                        self.maximum_delta_weights[j] = abs(delta_weights[j])

                delta_b = self.alpha * self.bias * (y_train[k] - self.net_input)

            for j in range(0, len(record)):
                self.weights[j] += delta_weights[j]

            self.b += delta_b

            for j in range(0, len(record)):
                if self.maximum_delta_weights[j] < self.threshold:
                    print('Breaking out of the algorithm...')
                    self.condition = True
                    break
        

    def predict(self, x_test, y_test):
        x_test = x_test
        y_test = y_test
        results = []
        for k in range(0, len(x_test)):
            record = x_test[k]

            for j in range(0, len(record)):
                self.x[j] = record[j]

            weighted_sum = 0
            for j in range(0, len(record)):
                weighted_sum += self.weights[j] * self.x[j]

            self.net_input = self.b + weighted_sum

            if self.net_input >= 0:
                self.output = 1
            else:
                self.output = -1

            results.append(self.output)

        show_plot(k+1, self.b, self.weights)  
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

'''
Q: why am i feeding the same data i fed to fit method, to the predict method?
A:
1. well, since i have four rows/records, and it's very few, it wasn't worth it to do a split for this homework.
2. since this is a "implementing an AND logic gate" problem which needs a plot (which i draw only one after everything
is finished), i just wanted to be faithful to my previous versions of perceptron and hebb.
so, it doesn't mean i worked less. i just wanted to show the decision line better and show that my implementation works.
'''

model = Adaline()
model.fit(X, labels)
predictions = model.predict(X, labels)
model.score(labels, predictions)
