# Project Batman, done by Alireza Rashidi Laleh
# AND logic gate done using perceptron
# python 3.7.8


import numpy as np
import matplotlib.pyplot as plt


#records = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [-1, 1, 1, -1], [-1, -1, 1, -1]])

# x(i) = s(i)
X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
labels = np.array([1, -1, -1, -1])
labels_colors = [-1, -1, -1, 1]

dummy_x1 = np.array([-2, 2])
dummy_x2 = np.array([0, 0])
dummy_x22 = np.array([0, 0])

xx1 = [-1, -1, 1, 1]
xx2 = [-1, 1, -1, 1]


def show_plot(j, next_b, next_weights, theta):
    for i in range(0, 2):
        print(next_weights[0])
        print(next_weights[1])
        if next_weights[0] == 0.0 and next_weights[1] == 0.0:
            dummy_x2[i] = 0
            dummy_x22[i] = 0
            #print(next_weights[0])
            #print(next_weights[1])
        else:
            dummy_x2[i] = float(-next_b - (next_weights[0] * dummy_x1[i]) + theta) / next_weights[1]
            #print(dummy_x2[i])
            dummy_x22[i] = float(-next_b - (next_weights[0] * dummy_x1[i]) - theta) / next_weights[1]

        #dummy_x1[i] = (-next_b - (next_weights[1] * dummy_x2[i]) + theta) / next_weights[0]
        #dummy_x22[i] = (-next_b - (next_weights[0] * dummy_x1[i]) - theta) / next_weights[1]


    fig = plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    plt.scatter(xx1, xx2, c=labels_colors, label=labels, linewidth=6)
    plt.plot(dummy_x1, dummy_x2, linewidth=5, color='r')
    plt.plot(dummy_x1, dummy_x22, linewidth=5, color='b')
    plt.title(f'Plot after reading record {j}')
    plt.show()


def perceptron(X, labels):
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
        alpha = 0.1
        theta = 0
        #alpha = 1
        #theta = 0.2

        #print(weights)

        x = {}
        for j in range(0, len(record)):
            x[j] = record[j]

        #print(x)

        weighted_sum = 0
        for j in range(0, len(record)):
            weighted_sum += weights[j] * x[j]

        net_input = b + weighted_sum
        #print(net_input)

        if net_input > theta:
            exit_neuron = 1
        elif net_input <= theta and net_input >= -theta:
            exit_neuron = 0
        elif net_input < -theta:
            exit_neuron = -1

        #print(exit_neuron)
        #print(labels[k])

        if exit_neuron != labels[k]:
            delta_weights = {}
            for j in range(0, len(record)):
                delta_weights[j] = alpha * x[j] * labels[k]

            next_weights = {}
            for j in range(0, len(record)):
                next_weights[j] = weights[j] + delta_weights[j]

            delta_b = alpha * bias * labels[k]
            next_b = b + delta_b
        else:
            next_neights = weights
            next_b = b

        if next_weights == weights:
            print('Exiting the algorithm...')
            #break

        show_plot(k+1, next_b, next_weights, theta)

perceptron(X, labels)
