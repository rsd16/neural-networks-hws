# Project Antiself, done by Alireza Rashidi Laleh
# AND logic gate done using Hebb Network
# python 3.7.8


import matplotlib.pyplot as plt
import numpy as np


records = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [-1, 1, 1, -1], [-1, -1, 1, -1]])

dummy_x1 = np.array([-2, 2])
dummy_x2 = np.array([0, 0])

xx1 = [-1, -1, 1, 1]
xx2 = [-1, 1, -1, 1]
labels = [-1, -1, -1, 1]


def show_plot(j, next_b, next_weight1, next_weight2):
    for i in range(0, 2):
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


def hebb_network(records):
    for i in range(0, 4):
        record = records[i]
        if not i == 0:
            weight1 = next_iteration[0]
            weight2 = next_iteration[1]
            b = next_iteration[2]
        else:
            weight1 = 0
            weight2 = 0
            b = 0

        x1 = record[0]
        x2 = record[1]
        bias = record[2]
        y = record[3]

        delta_weights1 = x1 * y
        delta_weights2 = x2 * y
        delta_b = bias * y

        next_weight1 = weight1 + delta_weights1
        next_weight2 = weight2 + delta_weights2
        next_b = b + delta_b

        next_iteration = [next_weight1, next_weight2, next_b]
        #net_input = next_b + next_weight1 * x1 + next_weight2 * x2

        show_plot(i+1, next_b, next_weight1, next_weight2)


hebb_network(records)
