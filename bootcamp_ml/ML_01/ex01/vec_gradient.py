import numpy as np


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.arrays, without any for loop.

    The three arrays must have compatible shapes.

    Args:
        x: has to be a numpy.array, a vector of shape m * 1.
        y: has to be a numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        None if x, y, or theta is an empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    if (not isinstance(y, np.ndarray)
            or not isinstance(x, np.ndarray)
            or not y.shape == x.shape
            or not isinstance(theta, np.ndarray)
            or not theta.shape == (2, 1)
            or y.size == 0
            or x.size == 0
            or theta.size == 0):
        return None
    m = x.size

    def j_param(x: np.ndarray, param: np.ndarray):
        X = np.hstack((np.ones((x.shape[0], 1)), x))
        y_hat = np.dot(X, param)
        return y_hat

    def deriv_J0(y_hat: np.ndarray, y: np.ndarray, m):
        derivative_J0 = np.array(y_hat - y).sum() / m
        return derivative_J0

    def deriv_J1(y_hat: np.ndarray, y: np.ndarray, x: np.ndarray, m):
        values = (y_hat - y) * x
        derivative_J1 = values.sum() / m
        return derivative_J1

    y_hat = j_param(x, theta)
    grad_0 = deriv_J0(y_hat, y, m)
    grad_1 = deriv_J1(y_hat, y, x, m)
    return np.array([grad_0, grad_1]).reshape(-1, 1)


if __name__ == "__main__":
    x = np.array([12.4956442, 21.5007972, 31.5527382,
                 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287,
                 46.6793434, 59.5585554]).reshape((-1, 1))
    # Example 0:
    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    res = gradient(x, y, theta1)
    print(res)
    # Output:
    # array([[-19.0342...], [-586.6687...]])

    # Example 1:
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    res = gradient(x, y, theta2)
    print(res)
    # Output:
    # array([[-57.8682...], [-2230.1229...]])
