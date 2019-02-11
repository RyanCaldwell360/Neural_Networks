import numpy as np

class linear_regressor:
    def __init__(self, X, Y, n_iters, learning_rate):
        self.X = X
        self.Y = Y
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.p = X.shape[0]
        self.N = X.shape[1]

    def initialize_weights(self):
        W = np.random.randn(self.p, 1)
        b = 0

        return W, b

    def forward_propogation(self, W, b):
        Z = np.dot(W.T, self.X) + b

        return Z

    def calculate_cost(self, y_hat):
        l = np.square(self.Y - y_hat)
        L = (1/self.N)*np.sum(l)

        return L

    def backward_propogation(self, y_hat):
        dW = (1/self.N)*np.dot(self.X, (y_hat - self.Y).T)
        db = (1/self.N)*np.sum((y_hat - self.Y), axis=1, keepdims=True)

        return dW, db

    def update_weights(self, old_W, old_b):
        W = W - self.learning_rate*dW
        b = b - self.learning_rate*db

        return W, b

    def train(self):
        W, b = initialize_weights()

        for i in range(self.n_iters):
            y_hat = forward_propogation(W, b)
            cost = calculate_cost(y_hat)
            bW, bb = backward_propogation(y_hat)
            W, b = update_weights(bW, bb)

            return W, b

    def predict(self, X_train, W, b):
        y_hat = np.dot(W.T, X_train) + b

        return y_hat
