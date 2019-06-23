import numpy as np
import codecs, json as JSON
from scipy.special import expit

def sigmoid(x):
    return 1 / (1 + np.e ** -x)

class RBFNet:

    def __init__(self, problem):
        self.X = None # input data matrix, each row is an input
        self.y = None # actual labels/outputs vector
        self.V = None # Circle centers, each row is a center
        self.G = None # first layer matrix
        self.W = None # weights from first layer to output layer
        self.y_hat = None # RBF Network output
        self.gamma = None
        self.problem = problem.lower()
        self.classes = None # number of classes

    # this method sets the X matrix and y vector for our network
    def set_inputs(self, X, y):
        self.X = X
        if self.problem == 'regression':
            self.y = y
        else:
            self.classes = int(np.max(y.ravel()) + 1)
            self.y = np.zeros((self.X.shape[0], self.classes))
            y = list(y.ravel())
            for i in range(X.shape[0]):
                col = int(y[i])
                self.y[i][col] = 1

    # this method is called for each individual (outputs of ES algorithm) before calculating the error
    def set_characteristics(self, V, gamma):
        self.V = V
        self.gamma = np.abs(gamma)

    # this method, fist calculates y_hat and then according to the problem, returns the error
    def update_network(self):
        self.G = self.compute_G()
        self.W = self.compute_W()
        self.y_hat = self.compute_y_hat()
        if self.problem == 'regression':
            return self.regression_error()
        elif self.problem == 'classification':
            return self.accuracy_error() + self.cross_entropy_error()

    # specific error function for regression problems
    def regression_error(self):
        return 0.5 * np.matmul((self.y_hat - self.y).T, self.y_hat - self.y)

    # specific error function for classification problems
    def cross_entropy_error(self):
        ans = 0
        for i in range(self.y.shape[0]):
            l = np.log(self.y_hat[i].T)
            ans += np.matmul(self.y[i], l)
            ans += np.matmul((1 - self.y[i]).T, np.log(1- self.y_hat[i]))
        ans *= (-1) / self.X.shape[0]
        return ans

    def accuracy_error(self):
        dataset_size = self.X.shape[0]
        incorrects = 0
        for i in range(dataset_size): 
            predicted_class = np.argmax(self.y_hat[i])
            actual_class = np.argmax(self.y[i])
            if predicted_class != actual_class: 
                incorrects += 1
        return np.array([[100 * incorrects / dataset_size]])


    # G matrix = result of applying inputs to radial basis functions
    def compute_G(self):
        res = []
        for i in range(self.X.shape[0]):
            row = []
            x_vec = self.X[i]
            for j in range(self.V.shape[0]):
                v_vec = self.V[j]
                sub = x_vec - v_vec
                power = - self.gamma[j][0] * np.matmul(sub, sub.T)
                row.append(np.exp(power))

            res.append(row)
        return np.array(res)

    # W matrix = weights
    def compute_W(self):
        GTG = np.matmul(self.G.T, self.G)
        GTGi = np.linalg.pinv(GTG)
        GTGiGT = np.matmul(GTGi, self.G.T)
        W = np.matmul(GTGiGT, self.y)
        return W

    # y_hat = our network's answer to X
    def compute_y_hat(self):
        y_hat = np.matmul(self.G, self.W)
        if self.problem == 'classification':
            y_hat = sigmoid(y_hat)

        return y_hat

    # this function saves W matrix in a .JSON file
    def save_results(self):
        file_path = "latest_train.json"  ## your path variable
        JSON.dump(self.W.tolist(), codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

    # this function applies our network to any given input vector
    def apply_network(self, x):
        x = np.array(x)
        g = []
        for i in range(self.V.shape[0]):
            v_vec = self.V[i]
            sub = x - v_vec
            power = - self.gamma[i][0] * np.matmul(sub, sub.T)
            g.append(np.exp(power))
        g = np.array(g)
        y = np.matmul(g, self.W)
        if self.problem == 'classification':
            y = sigmoid(y)
        return y