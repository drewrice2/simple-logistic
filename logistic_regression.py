import numpy as np

# ------------------------------------------------------------------ #
# This implentation of Logistic Regression is a programming          #
#   assignment from Coursera's Deep Learning course. It is for       #
#   binary classification. We use Logistic Regression as a starting  #
#   point for coding up neural networks.                             #
#                                                                    #
# There is definitely room to improve the overall organization of    #
#   the code. Perhaps, allowing for variable initialization to occur #
#   in `__init__`.                                                   #
# ------------------------------------------------------------------ #

class LogisticRegression:

    def __init__(self):
        """
        Variable initialization.
        """
        pass

    def sigmoid(self, z):
        """
        Compute the sigmoid of z.
        """
        s = 1 / (1 + np.exp(-z))
        return s

    def initialize_with_zeros(self, dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
        """
        w = np.zeros([dim, 1])
        b = 0

        # Coursera / Andrew Ng recommends adding plenty of `asserts` to code.
        # This helps check and validate matrix dimensions.
        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))
        return w, b

    def propagate(self, w, b, X, Y):
        """
        Implement the cost function and its gradient.
        """

        m = X.shape[1]
        # compute activation
        A = self.sigmoid(np.dot(w.T, X) + b)
        # compute cost
        cost = -(1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

        dw = (1/m)*np.dot(X,(A-Y).T)
        db = (1/m)*np.sum(A-Y)

        # Attempting to use more `assert` statements...
        assert(dw.shape == w.shape)
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        grads = {"dw": dw, "db": db}

        return grads, cost

    def optimize(self, w, b, X, Y, num_iterations, learning_rate):
        """
        This function optimizes w and b by running a gradient descent algorithm.
        """
        costs = []

        for i in range(num_iterations):
            # Compute gradient, cost.
            grads, cost = self.propagate(w, b, X, Y)

            dw = grads["dw"]
            db = grads["db"]

            # Perform updates.
            w = w - learning_rate * dw
            b = b - learning_rate * db

            # Record costs.
            if i % 100 == 0:
                costs.append(cost)

        params = {"w": w, "b": b}
        grads = {"dw": dw, "db": db}

        return params, grads, costs

    def predict(self, w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        '''

        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = w.reshape(X.shape[0], 1)

        # Compute probability of "True".
        A = self.sigmoid(np.dot(w.T, X) + b)
        # Rounding probabilities up or down. May not be the most efficient way to do this.
        Y_prediction = np.where(A>=0.5,1,0)
        # More `assert` statements.
        assert(Y_prediction.shape == (1, m))
        return Y_prediction

    def model(self, X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5):
        """
        Builds the logistic regression model.
        """
        # Initialize params.
        w, b = self.initialize_with_zeros(X_train.shape[0])
        # Perform gradient descent.
        parameters, grads, costs = self.optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
        w = parameters["w"]
        b = parameters["b"]
        Y_prediction_test = self.predict(w, b, X_test)
        Y_prediction_train = self.predict(w, b, X_train)

        # Print train/test Errors
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test,
             "Y_prediction_train" : Y_prediction_train,
             "w" : w,
             "b" : b,
             "learning_rate" : learning_rate,
             "num_iterations": num_iterations}

        return d
