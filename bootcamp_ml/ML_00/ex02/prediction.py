import numpy as np


def simple_predict(x: np.ndarray, theta: np.ndarray):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a one-dimensional array of size m.
        theta: has to be an numpy.ndarray, a one-dimensional array of size 2.
    Returns:
        y_hat as a numpy.ndarray, a one-dimensional array of size m.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    print(theta.shape)
    if len(x) == 0 or len(theta) == 0 or not theta.shape[0] == 2:
        return None
    m = len(x)
    y_hat = np.array(theta[0] + theta[1] * x)
    return y_hat


if __name__ == "__main__":
    x = np.arange(1, 6)
    # Example 1:
    theta1 = np.array([5, 0])
    res = simple_predict(x, theta1)
    print(res)
    # Ouput:
    # array([5., 5., 5., 5., 5.])
    # Do you understand why y_hat contains only 5s here?
    # Example 2:
    theta2 = np.array([0, 1])
    res = simple_predict(x, theta2)
    print(res)
    # Output:
    # array([1., 2., 3., 4., 5.])
    # Do you understand why y_hat == x here?
    # Example 3:
    theta3 = np.array([5, 3])
    res = simple_predict(x, theta3)
    print(res)
    # Output:
    # array([8., 11., 14., 17., 20.])
    # Example 4:
    theta4 = np.array([-3, 1])
    res = simple_predict(x, theta4)
    print(res)
    # Output:
    # array([-2., -1., 0., 1., 2.])
