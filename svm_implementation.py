import itertools
import numpy as np
import pandas as pd
from time import time
import cvxopt.solvers
import numpy.linalg as la
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5

class Kernel:
    @staticmethod
    def linear():
        return lambda x, y: np.dot(x.T, y)

    @staticmethod
    def polykernel(dimension, offset):
        return lambda x, y: (offset + np.dot(x.T, y)) ** dimension

    @staticmethod
    def radial_basis(gamma):
        return lambda x, y: np.exp(-gamma * la.norm(np.subtract(x, y)))

class SVMTrainer:
    def __init__(self, kernel, c=1.0):
        self.kernel = kernel
        self.c = c

    def compute_multipliers(self, X, y):
        n_samples, n_features = X.shape
        K = self.kernel_matrix(X, n_samples)

        # Add small regularization term to the kernel matrix
        P = cvxopt.matrix(np.outer(y, y) * K + np.eye(n_samples) * 1e-5)
        
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self.c)
        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))
        A = cvxopt.matrix(y, (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)

        # Attempt solving with CVXOPT, catching any ValueErrors
        try:
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)
            return np.ravel(solution['x'])
        except ValueError as e:
            print("Solver failed due to:", e)
            return np.zeros(n_samples)

    def train(self, X, y):
        lagrange_multipliers = self.compute_multipliers(X, y)
        return self.construct_predictor(X, y, lagrange_multipliers)

    def kernel_matrix(self, X, n_samples):
        # Compute the kernel matrix K (depends on the type of kernel used)
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])
        return K

    def construct_predictor(self, X, y, lagrange_multipliers):
        # Code to create a predictor using lagrange multipliers and support vectors
        support_vectors = lagrange_multipliers > 1e-5
        self.support_vectors_ = X[support_vectors]
        self.support_multipliers_ = lagrange_multipliers[support_vectors]
        self.support_labels_ = y[support_vectors]
        bias = np.mean([
            y_k - SVMPredictor(
                kernel=self.kernel,
                bias=0.0,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(x_k)
            for (y_k, x_k) in zip(support_vector_labels, support_vectors)
        ])
        return SVMPredictor(
            kernel=self.kernel,
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)



class SVMPredictor:
    def __init__(self, kernel, bias, weights, support_vectors, support_vector_labels):
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels

    def predict(self, x):
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights, self._support_vectors, self._support_vector_labels):
            result += z_i * y_i * self._kernel(x_i, x)
        return np.sign(result).item()

def calculate(true_positive, false_positive, false_negative, true_negative):
    result = {
        'precision': true_positive / (true_positive + false_positive),
        'recall': true_positive / (true_positive + false_negative)
    }
    return result

def confusion_matrix(true_positive, false_positive, false_negative, true_negative):
    matrix = PrettyTable([' ', 'Ham', 'Spam'])
    matrix.add_row(['Ham', true_positive, false_positive])
    matrix.add_row(['Spam', false_negative, true_negative])
    return matrix, calculate(true_positive, false_positive, false_negative, true_negative)

def write_to_file(matrix, result, parameters, kernel_type, start_time):
    with open("OneDrive/Desktop/IBM project/results/results.txt", "a") as f:
        f.write(f"Matrix: {matrix}\n")
        f.write(f"Result: {result}\n")
        f.write(f"Parameters: {parameters}\n")
        f.write(f"Type: {kernel_type}\n")
        f.write(f"Time for execution: {time() - start_time}\n\n")

def implementSVM(X_train, Y_train, X_test, Y_test, parameters, kernel_type):
    ham_spam = 0
    spam_spam = 0
    ham_ham = 0
    spam_ham = 0

    if kernel_type == "polykernel":
        dimension = parameters['dimension']
        offset = parameters['offset']
        trainer = SVMTrainer(Kernel.polykernel(dimension, offset), 0.1)
    elif kernel_type == "linear":
        trainer = SVMTrainer(Kernel.linear(), 0.1)
    predictor = trainer.train(X_train, Y_train)

    for i in range(X_test.shape[0]):
        ans = predictor.predict(X_test[i])
        if ans == -1 and Y_test[i] == -1:
            spam_spam += 1
        elif ans == 1 and Y_test[i] == -1:
            spam_ham += 1
        elif ans == 1 and Y_test[i] == 1:
            ham_ham += 1
        elif ans == -1 and Y_test[i] == 1:
            ham_spam += 1

    return confusion_matrix(ham_ham, ham_spam, spam_ham, spam_spam)

# Loading and cleaning data
df2 = pd.read_csv('OneDrive/Desktop/IBM project/results/frequency.csv', header=0)
input_output = np.nan_to_num(df2.to_numpy())  # Handle NaNs if present
X = input_output[:, :-1]
Y = input_output[:, -1:].ravel()  # Flatten Y to 1D array

# Splitting data
total = X.shape[0]
train = int(X.shape[0] * 0.7)
X_train, X_test = X[:train, :], X[train:, :]
Y_train, Y_test = Y[:train], Y[train:]

# Initialize output file
with open("OneDrive/Desktop/IBM project/results/results.txt", "w") as f:
    pass

# Execute kernel types
parameters = {'dimension': 2, 'offset': 1}
kernel_types = {"1": "polykernel", "2": "linear"}
k = 0

for i in range(2, 4):
    for j in range(0, 10):
        start_time = time()
        parameters.update({'dimension': i, 'offset': j})
        matrix, result = implementSVM(X_train, Y_train, X_test, Y_test, parameters, kernel_types['1'])
        write_to_file(matrix, result, parameters, kernel_types['1'], start_time)
        k += 1
        print("Done:", k)

# Linear kernel testing
start_time = time()
matrix, result = implementSVM(X_train, Y_train, X_test, Y_test, parameters, kernel_types['2'])
write_to_file(matrix, result, parameters, kernel_types['2'], start_time)
k += 1
print("Done:", k)

# Write total time spent
with open("OneDrive/Desktop/IBM project/results/results.txt", "a") as f:
    f.write("Total time spent: " + str(round(time() - start_time, 2)) + "\n")
