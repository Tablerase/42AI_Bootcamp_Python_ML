import numpy as np


def loss_(y: np.ndarray, y_hat: np.ndarray):
    """Computes the half mean-squared-error of two non-empty numpy.arrays, without any for loop.

    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a one-dimensional array of size m.
        y_hat: has to be an numpy.array, a one-dimensional array of size m.
    Returns:
        The half mean-squared-error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
    Raises:
        This function should not raise any Exceptions.
    """
    if (not isinstance(y, np.ndarray)
            or not y.ndim == 1
            or not isinstance(y_hat, np.ndarray)
            or not y_hat.ndim == 1
            or not y.shape == y_hat.shape
            or y.size == 0):
        return None
    J_value = np.dot(y_hat - y, y_hat - y) / (2 * y.size)
    return J_value


if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    # Example 1:
    res = loss_(X, Y)
    print(res)
    # Output:
    # 2.142857142857143
    # Example 2:
    res = loss_(X, X)
    print(res)
    # Output:
    # 0.0
