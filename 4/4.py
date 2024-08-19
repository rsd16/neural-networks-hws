# Project Memories, done by Alireza Rashidi Laleh
# use perceptron for optical character recognition.
# python 3.7.8


import numpy as np


class Perceptron:
    def __init__(self, alpha=0.001, theta=0.0001):
        self.bias = 1
        self.alpha = alpha
        self.theta = theta
        self.x = {}
        self.weights = {}
        self.b = 0
        self.net_input = 0
        self.updated = False

    def fit(self, x_train, y_train):
        x_train = x_train
        y_train = y_train
        for k in range(0, len(x_train)):
            self.updated = False
            record = x_train[k]
            if k == 0:
                self.b = 0
                for j in range(0, len(record)):
                    self.weights[j] = 0

            self.x = {}
            for j in range(0, len(record)):
                self.x[j] = record[j]

            weighted_sum = 0
            for j in range(0, len(record)):
                weighted_sum += self.weights[j] * self.x[j]

            self.net_input = self.b + weighted_sum

            if self.net_input > self.theta:
                exit_neuron = 1
            elif self.net_input <= self.theta and self.net_input >= -self.theta:
                exit_neuron = 0
            elif self.net_input < -self.theta:
                exit_neuron = -1

            if exit_neuron != y_train[k]:
                self.updated = True
                delta_weights = {}
                for j in range(0, len(record)):
                    delta_weights[j] = self.alpha * self.x[j] * y_train[k]

                next_weights = {}
                for j in range(0, len(record)):
                    self.weights[j] += delta_weights[j]

                delta_b = self.alpha * self.bias * y_train[k]
                self.b += delta_b

            if not self.updated:
                print('Breaking out of the algorithm...')
                #break
                

    def predict(self, x_test, y_test):
        x_test = x_test
        y_test = y_test
        results = []
        for k in range(0, len(x_test)):
            record = x_test[k]
            
            self.x = {}
            for j in range(0, len(record)):
                self.x[j] = record[j]

            weighted_sum = 0
            for j in range(0, len(record)):
                weighted_sum += self.weights[j] * self.x[j]

            self.net_input = self.b + weighted_sum
            #print(net_input)

            if self.net_input > self.theta:
                exit_neuron = 1
            elif self.net_input <= self.theta and self.net_input >= -self.theta:
                exit_neuron = 0
            elif self.net_input < -self.theta:
                exit_neuron = -1

            results.append(exit_neuron)

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

model = Perceptron()
#print(model)
#model.alpha = 5
#print(model.alpha)

model.fit(x_train, y_train)
predictions = model.predict(x_test, y_test)

print('Actual targets:')
print(y_test)
print('Predicted Targets:')
print(predictions)

accuracy = model.score(y_test, predictions)
print(accuracy)
