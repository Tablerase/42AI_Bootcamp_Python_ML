import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot(x: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a one-dimensional array of size m.
        y: has to be an numpy.array, a one-dimensional array of size m.
        theta: has to be an numpy.array, a two-dimensional array of shape 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exceptions.
    """
    if (not isinstance(x, np.ndarray)
            or not x.ndim == 1
            or not isinstance(y, np.ndarray)
            or not y.ndim == 1
            or not isinstance(theta, np.ndarray)
            or not theta.shape == (2, 1)):
        return
    X = x[:, np.newaxis]
    H = np.hstack((np.ones((x.shape[0], 1)), X))
    y_hat = np.dot(H, theta)

    plt.scatter(x, y, marker='o', alpha=0.6, label=f"data points")
    plt.plot(x, y_hat, color='red', label=f"prediction line")
    plt.show()


if __name__ == "__main__":
    x = np.arange(1, 6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])

    # Example 1:
    theta1 = np.array([[4.5], [-0.2]])
    plot(x, y, theta1)

    # Example 2:
    theta2 = np.array([[-1.5], [2]])
    plot(x, y, theta2)

    # Example 3:
    theta3 = np.array([[3], [0.3]])
    plot(x, y, theta3)
