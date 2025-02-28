import numpy as np


def gradient(x, y, theta):
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
    y_hat = np.dot(X_bias, theta)
    diff_error = y_hat - y
    gradient_vect = np.dot(X_bias.T, diff_error) / m
    return gradient_vect


def fit_(x: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha, max_iter):
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
    if (not isinstance(y, np.ndarray)
            or not y.shape == (m, 1)
            or not isinstance(x, np.ndarray)
            or not x.shape == (m, n)
            or not isinstance(theta, np.ndarray)
            or not theta.shape == (n + 1, 1)
            or y.size == 0
            or x.size == 0
            or theta.size == 0
            or not isinstance(alpha, float)
            or not isinstance(max_iter, int)):
        return None

    for i in range(max_iter):
        if i % 1000 == 0:
            print(f"{i}/{max_iter}", end='\r')
        gradient_J = gradient(x, y, theta)
        theta = theta - alpha * gradient_J
        if np.isnan(gradient_J).any() or gradient_J.max() > 1e10 or gradient_J.min() < -1e10:
            break
        if gradient_J.any() == 0:
            break
    return theta


if __name__ == "__main__":
    import numpy as np
    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.],
                 [0.6, 6., 60.], [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])

    # Example 0:
    theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
    res = theta2
    print(res)
    # Output:
    # array([[41.99..], [0.97..], [0.77..], [-1.20..]])

    # Example 1:
    # res = predict_(x, theta2)
    print(res)
    # Output:
    # array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])
