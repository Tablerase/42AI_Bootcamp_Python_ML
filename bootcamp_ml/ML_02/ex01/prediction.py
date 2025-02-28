import numpy as np


def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.

    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
    Return:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not matching.
        None if x or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    m = x.shape[0]
    n = x.shape[1]
    if (not isinstance(x, np.ndarray)
            or not x.ndim >= 2
            or not isinstance(theta, np.ndarray)
            or not theta.shape == ((n + 1), 1)):
        return None
    X = np.hstack((np.ones((m, 1)), x))
    y_hat = np.dot(X, theta)
    return y_hat


if __name__ == "__main__":
    x = np.arange(1, 13).reshape((4, -1))
    print("x:", x)
    # Example 1:
    theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
    res = predict_(x, theta1)
    print("Example 1:", res)
    # Ouput:
    # array([[5.], [5.], [5.], [5.]])
    # Do you understand why y_hat contains only 5’s here?
    # Only b is set other are x * 0

    # Example 2:
    theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
    res = predict_(x, theta2)
    print("Example 2:", res)
    # Output:
    # array([[1.], [4.], [7.], [10.]])
    # Do you understand why y_hat == x[:,0] here?
    # Only w1 is set and other are 0 + w1*x + 0*x + ...

    # Example 3:
    theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
    res = predict_(x, theta3)
    print("Example 3:", res)
    # Output:
    # array([[9.64], [24.28], [38.92], [53.56]])

    # Example 4:
    theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
    res = predict_(x, theta4)
    print("Example 4:", res)
    # Output:
    # array([[12.5], [32.], [51.5], [71.]])
