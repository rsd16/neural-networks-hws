# Project Death, done by Alireza Rashidi Laleh
# use hebb network for optical character recognition.
# python 3.7.8


import matplotlib.pyplot as plt
import numpy as np


records = np.loadtxt('preprocessd data.txt')
#print(records.shape)

X = records[:, 0:-1]
#print(X.shape)
#print(X)
labels = records[:, -1]
#print(labels.shape)
#print(labels)


def show_plot(j, next_b, next_weight1, next_weight2):
    for i in range(0, 5):
        dummy_x2[i] = (-next_b - (next_weight1 * dummy_x1[i])) / next_weight2

    fig = plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    plt.scatter(xx1, xx2, c=labels, label=labels, linewidth=6)
    plt.plot(dummy_x1, dummy_x2, linewidth=5, color='r')
    plt.title(f'Plot after reading record {j}')
    plt.show()


def hebb_network(X, labels):
    for k in range(0, len(X)):
        record = X[k]
        #print(record)
        weights = {}
        if not k == 0:
            b = next_b
            for j in range(0, len(record)):
                weights[j] = next_weights[j]
        else:
            b = 0
            for j in range(0, len(record)):
                weights[j] = 0

        bias = 1
        print(weights)

        x = {}
        for j in range(0, len(record)):
            x[j] = record[j]

        #print(x)

        delta_weights = {}
        for j in range(0, len(record)):
            delta_weights[j] = x[j] * labels[k]

        #print(delta_weights)

        next_weights = {}
        for j in range(0, len(record)):
            next_weights[j] = weights[j] + delta_weights[j]

        #print(next_weights)

        delta_b = bias * labels[k]
        next_b = b + delta_b
        #print(next_b)

        weighted_sum = 0
        for j in range(0, len(record)):
            weighted_sum += next_weights[j] * x[j]

        #print(weighted_sum)

        net_input = next_b + weighted_sum
        print(net_input)

        #show_plot(i+1, next_b, next_weight1, next_weight2)


hebb_network(X, labels)
