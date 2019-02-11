import numpy as np

class D1NN:
    def __init__(self, n_iters, n_nodes, learning_rate):
        self.n_iters = n_iters
        self.n_nodes = n_nodes
        self.learning_rate = learning_rate

    def initialize_weights(self, feature_size):
        W1 = np.random.randn(self.n_nodes, feature_size)*np.sqrt(2/feature_size)
        W2 = np.random.randn(1, self.n_nodes)*np.sqrt(2/feature_size)
        b1 = np.random.uniform(0, 1, self.n_nodes).reshape(self.n_nodes, 1)
        b2 = np.random.uniform(0, 1, 1).reshape(1, 1)

        weights = {'W1':W1,
                   'b1':b1,
                   'W2':W2,
                   'b2':b2}

        return weights

    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def forward_propogation(self, params, X):
        z1 = np.dot(params['W1'], X) + params['b1']
        a1 = self.relu(z1)
        z2 = np.dot(params['W2'], a1) + params['b2']
        a2 = self.sigmoid(z2)

        fp = {'z1':z1,
              'a1':a1,
              'z2':z2,
              'a2':a2}

        return fp

    def backward_propogation(self, weights, forward_passes, X, Y):
        N = X.shape[1]

        dZ2 = forward_passes['a2'] - Y
        dZ1 = np.dot(weights['W2'].T, dZ2)*(1 - np.power(forward_passes['a1'], 2))

        dW2 = (1/N)*(np.dot(dZ2, forward_passes['a1'].T))
        dW1 = (1/N)*(np.dot(dZ1, X.T))

        db2 = (1/N)*(np.sum(dZ2, axis=1, keepdims=True))
        db1 = (1/N)*(np.sum(dZ1, axis=1, keepdims=True))

        gradients = {'dW1':dW1,
                     'db1':db1,
                     'dW2':dW2,
                     'db2':db2}

        return gradients

    def update_weights(self, gradients, weights):
        W1 = weights['W1'] - self.learning_rate*gradients['dW1']
        b1 = weights['b1'] - self.learning_rate*gradients['db1']

        W2 = weights['W2'] - self.learning_rate*gradients['dW2']
        b2 = weights['b2'] - self.learning_rate*gradients['db2']

        weights = {'W1':W1,
                   'b1':b1,
                   'W2':W2,
                   'b2':b2}

        return weights

    def calculate_cost(self, params, Y):
        N = Y.shape[1]
        a2 = params['a2']

        return -(1/N)*np.sum(np.multiply(Y,np.log(a2)) + np.multiply((1-Y),np.log(1-a2)))

    def train(self, X, Y):
        # initialize weights
        feature_size = X.shape[0]
        weights = self.initialize_weights(feature_size)

        # Number of iterations to update weights
        costs = list()
        for i in range(self.n_iters):
            # forward pass of data
            forward_pass = self.forward_propogation(weights, X)

            # calculate cost
            cost = self.calculate_cost(forward_pass, Y)
            costs.append(cost)

            # calculate gradients through backwards propogation
            grads = self.backward_propogation(weights, forward_pass, X, Y)

            # update weights
            weights = self.update_weights(grads, weights)

        return (weights, costs)

    def predict_proba(self, weights, X):
        fp = self.forward_propogation(weights, X)

        return 1/(1 + np.exp(-fp['a2']))
