import numpy as np


def loss_(y, y_hat):
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
    m = y.shape[0]
    if (not isinstance(y, np.ndarray)
            or not isinstance(y_hat, np.ndarray)):
        return None
    J_elem = np.array((y - y_hat)**2)
    Jwb = J_elem.sum() / (2 * y.size)
    return Jwb


if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    # Example 1:
    res = loss_(X, Y)
    print("Example 1:", res)
    # Output:
    # 2.142857142857143

    # Example 2:
    res = loss_(X, X)
    print("Example 2:", res)
    # Output:
    # 0.0
