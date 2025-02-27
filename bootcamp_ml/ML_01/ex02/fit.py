import numpy as np
from bootcamp_ml.ML_00.ex06.loss import predict_


def fit_(x, y, theta, alpha, max_iter):
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
            or not isinstance(theta, np.ndarray)
            or not theta.shape == (2, 1)
            or y.size == 0
            or x.size == 0
            or theta.size == 0
            or not isinstance(alpha, float)
            or not isinstance(max_iter, int)):
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

    step_size = alpha
    limit = max_iter
    param = theta
    for i in range(limit):
        # if i % 1000 == 0:
        #     print(i)
        dJ0 = deriv_J0(j_param(x, param), y, m)
        dJ1 = deriv_J1(j_param(x, param), y, x, m)
        tmp_w = param[0][0] - step_size * dJ0
        tmp_b = param[1][0] - step_size * dJ1
        param = np.array([tmp_w, tmp_b]).reshape(-1, 1)
        if np.isnan(dJ0) or np.isnan(dJ1) or np.abs(dJ0) > 1e10 or np.abs(dJ1) > 1e10:
            break
        if dJ0 == 0 or dJ1 == 0:
            break
    return param


if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [
                 31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [
                 45.7655287], [46.6793434], [59.5585554]])
    theta = np.array([1, 1]).reshape((-1, 1))

    # Example 0:
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    res = theta1
    print(res)
    # Output:
    # array([[1.40709365],
    # [1.1150909 ]])

    # Example 1:
    res = predict_(x, theta1)
    print(res)
    # Output:
    # array([[15.3408728 ],
    # [25.38243697],
    # [36.59126492],
    # [55.95130097],
    # [65.53471499]])
