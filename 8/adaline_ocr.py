# Project Pynchon, done by Alireza Rashidi Laleh
# Adaline network for optical character recognition.
# python 3.7.8


import numpy as np


class Adaline:
    def __init__(self, alpha=0.0001, threshold=0.001):
        self.bias = 1
        self.alpha = alpha
        self.threshold = threshold
        self.x = {}
        self.weights = {}
        self.b = 0
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
                    self.b = 0
                    for j in range(0, len(record)):
                        self.weights[j] = 0
                        self.maximum_delta_weights[j] = 0

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

                for j in range(0, len(record)):
                    self.weights[j] += delta_weights[j]

                delta_b = self.alpha * self.bias * (y_train[k] - self.net_input)
                self.b += delta_b

            max_difference = max(self.maximum_delta_weights.values())
            #print(max_difference)
            if max_difference < self.threshold:
                print('Breaking out of the algorithm...')
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

model = Adaline()
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
