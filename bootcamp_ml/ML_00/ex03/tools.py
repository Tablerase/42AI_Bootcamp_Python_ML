import numpy as np


def add_intercept(x: np.ndarray):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array. x can be a one-dimensional (m * 1) or two-dimensional (m * n) array.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or len(x) == 0:
        return None
    ones = np.ones((x.shape[0], 1))
    if x.ndim == 1:
        x = x[:, np.newaxis]  # np.transpose() not for 1d array
    new_matrix = np.hstack((ones, x))
    return new_matrix


if __name__ == "__main__":
    # Example 1:
    x = np.arange(1, 6)
    res = add_intercept(x)
    print(res)
    # Output:
    # array([[1., 1.],
    #        [1., 2.],
    #        [1., 3.],
    #        [1., 4.],
    #        [1., 5.]])
    # Example 2:
    y = np.arange(1, 10).reshape((3, 3))
    res = add_intercept(y)
    print(res)
    # Output:
    # array([[1., 1., 2., 3.],
    #        [1., 4., 5., 6.],
    #        [1., 7., 8., 9.]])
