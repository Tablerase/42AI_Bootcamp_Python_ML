import numpy as np
import matplotlib.pyplot as plt
# from bootcamp_ml.ML_00.ex04.prediction import predict_


def predict_(x: np.ndarray, theta: np.ndarray):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.

    Args:
        x: has to be an numpy.array, a one-dimensional array of size m.
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
            or not isinstance(theta, np.ndarray)
            #! or not x.ndim == 1
            or not theta.shape == (2, 1)):
        return None
    #! X = x[:, np.newaxis]
    M = np.hstack((np.ones((x.shape[0], 1)), x))
    y_hat = np.dot(M, theta)
    return y_hat


def loss_elem_(y: np.ndarray, y_hat: np.ndarray):
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


def loss_(y: np.ndarray, y_hat: np.ndarray):
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
    J_elem = loss_elem_(y, y_hat)
    m = y.shape[0]
    J_value = J_elem.sum() / (2 * m)
    return J_value


if __name__ == "__main__":
    x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])
    # ! x1 2 dim array reject by predict (ex04) that accept only 1 dim
    y_hat1 = predict_(x1, theta1)
    y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

    # Example 1:
    res = loss_elem_(y1, y_hat1)
    print(res)
    # Output:
    # array([[0.], [1], [4], [9], [16]])

    # Example 2:
    res = loss_(y1, y_hat1)
    print(res)
    # Output:
    # 3.0

    x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
    theta2 = np.array(np.array([[0.], [1.]]))
    y_hat2 = predict_(x2, theta2)
    y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)

    # Example 3:
    res = loss_(y2, y_hat2)
    print(res)
    # Output:
    # 2.142857142857143

    # Example 4:
    res = loss_(y2, y2)
    print(res)
    # Output:
    # 0.0
