import numpy as np


class MyLinearRegression:
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def gradient(self, x, y):
        """Computes a gradient vector from three non-empty numpy.array, without any for-loop.

        The three arrays must have the compatible dimensions.
        Args:
            x: has to be an numpy.array, a matrix of dimension m * n.
            y: has to be an numpy.array, a vector of dimension m * 1.
            theta: has to be an numpy.array, a vector (n +1) * 1.
        Return:
            The gradient as a numpy.array, a vector of dimensions n * 1,
            containg the result of the formula for all j.
            None if x, y, or theta are empty numpy.array.
            None if x, y and theta do not have compatible dimensions.
            None if x, y or theta is not of expected type.
        Raises:
            This function should not raise any Exception.
        """
        m = x.shape[0]
        n = x.shape[1]
        X_bias = np.hstack((np.ones((m, 1)), x))
        y_hat = np.dot(X_bias, self.thetas)
        diff_error = y_hat - y
        gradient_vect = np.dot(X_bias.T, diff_error) / m
        return gradient_vect

    def fit_(self, x: np.ndarray, y: np.ndarray):
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.array, a matrix of dimension m * n:
            (number of training examples, number of features).
            y: has to be a numpy.array, a vector of dimension m * 1:
            (number of training examples, 1).
            theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
            (number of features + 1, 1).
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during the gradient descent
        Return:
            new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
            None if there is a matching dimension problem.
            None if x, y, theta, alpha or max_iter is not of expected type.
        Raises:
            This function should not raise any Exception.
        """
        m = x.shape[0]
        n = x.shape[1]
        # if (not isinstance(y, np.ndarray)
        #         or not y.shape == (m, 1)
        #         or not isinstance(x, np.ndarray)
        #         or not x.shape == (m, n)
        #         or not isinstance(self.thetas, np.ndarray)
        #         or not self.thetas.shape == (n + 1, 1)
        #         or y.size == 0
        #         or x.size == 0
        #         or self.thetas.size == 0
        #         or not isinstance(self.alpha, float)
        #         or not isinstance(self.max_iter, int)):
        #     return None

        for i in range(self.max_iter):
            if i % 1000 == 0:
                print(f"{i}/{self.max_iter}", end='\r')
            gradient_J = self.gradient(x, y)
            self.thetas = self.thetas - self.alpha * gradient_J
            if np.isnan(gradient_J).any() or gradient_J.max() > 1e10 or gradient_J.min() < -1e10:
                break
            if gradient_J.any() == 0:
                break
        return self.thetas

    def loss_elem_(self, y, y_hat):
        res = np.array((y - y_hat)**2)
        return res

    def loss_(self, y, y_hat):
        """Computes the mean squared error of two non-empty numpy.array, without any for loop.

        The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Return:
            The mean squared error of the two vectors as a float.
            None if y or y_hat are empty numpy.array.
            None if y and y_hat does not share the same dimensions.
            None if y or y_hat is not of expected type.
        Raises:
            This function should not raise any Exception.
        """
        if (not isinstance(y, np.ndarray)
                or not isinstance(y_hat, np.ndarray)):
            return None
        J_elem = np.array((y - y_hat)**2)
        Jwb = J_elem.sum() / (2 * y.size)
        return Jwb

    def predict_(self, x):
        """Computes the prediction vector y_hat from two non-empty numpy.array.

        Args:
            x: has to be an numpy.array, a matrix of dimension m * n.
            theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
        Return:
            y_hat as a numpy.array, a vector of dimension m * 1.
            None if x or theta are empty numpy.array.
            None if x or theta dimensions are not matching.
            None if x or theta is not of expected type.
        Raises:
            This function should not raise any Exception.
        """
        m = x.shape[0]
        n = x.shape[1]
        # if (not isinstance(x, np.ndarray)
        # or not x.ndim >= 2
        # or not isinstance(self.thetas, np.ndarray)
        # or not self.thetas.shape == ((n + 1), 1)):
        # return None
        X = np.hstack((np.ones((m, 1)), x))
        y_hat = np.dot(X, self.thetas)
        return y_hat


if __name__ == "__main__":
    MyLR = MyLinearRegression
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyLR([[1.], [1.], [1.], [1.], [1]])

    # Example 0:
    y_hat = mylr.predict_(X)
    print("y_hat:", y_hat)
    # Output:
    # array([[8.], [48.], [323.]])

    # Example 1:
    res = mylr.loss_elem_(Y, y_hat)
    print("loss_elem:", res)
    # Output:
    # array([[225.], [0.], [11025.]])

    # Example 2:
    res = mylr.loss_(Y, y_hat)
    print("loss:", res)
    # Output:
    # 1875.0

    # Example 3:
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    res = mylr.fit_(X, Y)
    mylr.thetas
    print("thetas:", res)
    # Output:
    # array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])

    # Example 4:
    y_hat = mylr.predict_(X)
    print("y_hat:", y_hat)
    # Output:
    # array([[23.417..], [47.489..], [218.065...]])

    # Example 5:
    res = mylr.loss_elem_(Y, y_hat)
    print("loss_elem:", res)
    # Output:
    # array([[0.174..], [0.260..], [0.004..]])

    # Example 6:
    res = mylr.loss_(Y, y_hat)
    print("loss:", res)
    # Output:
    # 0.0732..
