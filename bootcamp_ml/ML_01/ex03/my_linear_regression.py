import numpy as np


class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def predict_(self, x: np.ndarray):
        """Computes the vector of prediction y_hat from two non-empty numpy.array.

        Args:
            x: has to be an numpy.array.
            theta: has to be an numpy.array, a two-dimensional array of shape 2 * 1.
        Returns:
            y_hat as a numpy.array, a two-dimensional array of shape m * 1.
            None if x and/or theta are not numpy.array.
            None if x or theta are empty numpy.array.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exceptions.
        """
        if (not isinstance(x, np.ndarray)
                or not isinstance(self.thetas, np.ndarray)
                #! or not x.ndim == 1
                or not self.thetas.shape == (2, 1)):
            return None
        #! X = x[:, np.newaxis]
        X = np.hstack((np.ones((x.shape[0], 1)), x))
        y_hat = np.dot(X, self.thetas)
        return y_hat

    def fit_(self, x, y):
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during the gradient descent
        Returns:
            new_theta: numpy.ndarray, a vector of dimension 2 * 1.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exception.
        """
        if (not isinstance(y, np.ndarray)
                or not isinstance(x, np.ndarray)
                or not y.shape == x.shape
                or not isinstance(self.thetas, np.ndarray)
                or not self.thetas.shape == (2, 1)
                or y.size == 0
                or x.size == 0
                or self.thetas.size == 0
                or not isinstance(self.alpha, float)
                or not isinstance(self.max_iter, int)):
            return None
        m = x.size

        def deriv_J0(y_hat: np.ndarray, y: np.ndarray, m):
            derivative_J0 = np.array(y_hat - y).sum() / m
            return derivative_J0

        def deriv_J1(y_hat: np.ndarray, y: np.ndarray, x: np.ndarray, m):
            values = (y_hat - y) * x
            derivative_J1 = values.sum() / m
            return derivative_J1

        step_size = self.alpha
        for i in range(self.max_iter):
            if i % 100 == 0:
                print(f'{i} / {self.max_iter}', end='\r')
            dJ0 = deriv_J0(self.predict_(x), y, m)
            dJ1 = deriv_J1(self.predict_(x), y, x, m)
            tmp_w = self.thetas[0][0] - step_size * dJ0
            tmp_b = self.thetas[1][0] - step_size * dJ1
            self.thetas = np.array([tmp_w, tmp_b]).reshape(-1, 1)
            if np.isnan(dJ0) or np.isnan(dJ1) or np.abs(dJ0) > 1e10 or np.abs(dJ1) > 1e10:
                break
            if dJ0 == 0 or dJ1 == 0:
                break
        return self.thetas

    def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray):
        """
        Description:
            Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
            y: has to be an numpy.array, a two-dimensional array of shape m * 1.
            y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
        Returns:
            J_elem: numpy.array, a array of dimension (number of the training examples, 1).
            None if there is a dimension matching problem.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if (not isinstance(y, np.ndarray)
                or not y.ndim == 2
                or not isinstance(y_hat, np.ndarray)
                or not y_hat.ndim == 2
                or not y.shape == y_hat.shape
                or not y.shape[1] == 1):
            return None
        J_elem = (y_hat - y)**2
        return J_elem

    def loss_(self, y: np.ndarray, y_hat: np.ndarray):
        """
        Description:
            Calculates the value of loss function.
        Args:
            y: has to be an numpy.array, a two-dimensional array of shape m * 1.
            y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
        Returns:
            J_value : has to be a float.
            None if there is a dimension matching problem.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if (not isinstance(y, np.ndarray)
                or not y.ndim == 2
                or not isinstance(y_hat, np.ndarray)
                or not y_hat.ndim == 2
                or not y.shape == y_hat.shape
                or not y.shape[1] == 1):
            return None
        J_elem = self.loss_elem_(y, y_hat)
        m = y.shape[0]
        J_value = J_elem.sum() / (2 * m)
        return J_value


if __name__ == "__main__":
    import numpy as np
    MyLR = MyLinearRegression
    x = np.array([[12.4956442], [21.5007972], [
                 31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [
                 45.7655287], [46.6793434], [59.5585554]])
    lr1 = MyLR(np.array([[2], [0.7]]))

    # Example 0.0:
    y_hat = lr1.predict_(x)
    print("0.0:", y_hat)
    # Output:
    # array([[10.74695094],
    #        [17.05055804],
    #        [24.08691674],
    #        [36.24020866],
    #        [42.25621131]])

    # Example 0.1:
    res = lr1.loss_elem_(y, y_hat)
    print("0.1:", res)
    # Output:
    # array([[710.45867381],
    #        [364.68645485],
    #        [469.96221651],
    #        [108.97553412],
    #        [299.37111101]])

    # Example 0.2:
    res = lr1.loss_(y, y_hat)
    print("0.2:", res)
    # Output:
    # 195.34539903032385

    # Example 1.0:
    lr2 = MyLR(np.array([[1], [1]]), 5e-8, 1500000)
    lr2.fit_(x, y)
    res = lr2.thetas
    print("1.0:", res)
    # Output:
    # array([[1.40709365],
    #    [1.1150909]])

    # Example 1.1:
    y_hat = lr2.predict_(x)
    print("1.2:", y_hat)
    # Output:
    # array([[15.3408728],
    #        [25.38243697],
    #        [36.59126492],
    #        [55.95130097],
    #        [65.53471499]])

    # Example 1.2:
    res = lr2.loss_elem_(y, y_hat)
    print("1.2:", res)
    # Output:
    # array([[486.66604863],
    #        [115.88278416],
    #        [84.16711596],
    #        [85.96919719],
    #        [35.71448348]])

    # Example 1.3:
    res = lr2.loss_(y, y_hat)
    print("1.3:", res)
    # Output:
    # 80.83996294128525
