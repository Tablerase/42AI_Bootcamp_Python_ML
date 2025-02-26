import numpy as np


def simple_gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """Computes a gradient vector from three non-empty numpy.arrays, with a for-loop.

    The three arrays must have compatible shapes.

    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
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

    # step_size = 0.0001
    # limit = 100
    # param = theta
    # for _ in range(limit):
    #     dJ0 = deriv_J0(j_param(x, param), y, m)
    #     dJ1 = deriv_J1(j_param(x, param), y, x, m)
    #     tmp_w = param[0][0] - step_size * dJ0
    #     tmp_b = param[1][0] - step_size * dJ1
    #     param = np.array([tmp_w, tmp_b]).reshape(-1, 1)
    #     if np.isnan(dJ0) or np.isnan(dJ1) or np.abs(dJ0) > 1e10 or np.abs(dJ1) > 1e10:
    #         break
    #     if dJ0 == 0 or dJ1 == 0:
    #         break
    # return param
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
    res = simple_gradient(x, y, theta1)
    print(res)
    # Output:
    # array([[-19.0342574], [-586.66875564]])

    # Example 1:
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    res = simple_gradient(x, y, theta2)
    print(res)
    # Output:
    # array([[-57.86823748], [-2230.12297889]])
