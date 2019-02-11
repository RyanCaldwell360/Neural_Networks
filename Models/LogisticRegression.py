import numpy as np

class LogisticRegression:
    def __init__(self, n_iters, learning_rate, random_weights=False):
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.random_weights = random_weights

    def init_weights(self, dims):
        if self.random_weights:
            w = np.random.uniform(0, 1, dims).reshape(dims, 1)
            b = np.random.uniform(0, 1, 1)[0]
        else:
            w = np.zeros((dims, 1))
            b = 0

        return w, b

    def _sigmoid(self, w, b, X):
        z = np.dot(w.T,X) + b

        return 1/(1 + np.exp(-z))

    def _cost(self, a, Y):
        dim = a.shape[0]
        return -(1/dim)*np.sum((Y*np.log(a)) + ((1 - Y)*np.log(1 - a)))

    def _updates(self, w, b, X, Y):
        dim = X.shape[1]

        a = self._sigmoid(w, b, X)
        J = self._cost(a, Y)

        dw = (1/dim)*(np.dot(X, (a - Y).T))
        db = (1/dim)*np.sum((a - Y), axis=1, keepdims=True)
        w = w - self.learning_rate*dw
        b = b - self.learning_rate*db

        return dw, db, w, b, J

    def train(self, X_train, Y_train):
        # initialize our weights
        dims = X_train.shape[0]
        w, b = self.init_weights(dims)

        dws = list()
        dbs = list()
        costs = list()
        for i in range(self.n_iters):
            # make pred, evaluate, back-prop, update weights
            dw, db, w, b, cost = self._updates(w, b, X_train, Y_train)
            dws.append(dw)
            dbs.append(db)
            costs.append(cost)

        return dws, dbs, w, b, costs

    def predict_proba(self, w, b, X):
        a = self._sigmoid(w, b, X)

        return a
