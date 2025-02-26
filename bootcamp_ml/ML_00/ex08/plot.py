import numpy as np
import matplotlib.pyplot as plt


def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.

    Args:
        x: has to be an numpy.ndarray, one-dimensional array of size m.
        y: has to be an numpy.ndarray, one-dimensional array of size m.
        theta: has to be an numpy.ndarray, one-dimensional array of size 2.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    """
    if (not isinstance(y, np.ndarray)
            or not y.ndim == 1
            or not isinstance(x, np.ndarray)
            or not x.ndim == 1
            or not y.shape == x.shape
            or not isinstance(theta, np.ndarray)
            or not theta.ndim == 1
            or not theta.size == 2):
        return
    # Predictions
    X = x[:, np.newaxis]
    X = np.hstack((np.ones((x.shape[0], 1)), X))
    y_hat = np.dot(X, theta)
    # Evaluations
    # ! Subject not clear if doing MSE or Cost because graphic cost in test is MSE
    MSE_value = np.dot(y_hat - y, y_hat - y) / (y.size)
    J_value = MSE_value / 2

    # Plotting
    plt.scatter(x, y, marker='o', label="data points")
    plt.plot(x, y_hat, color="red", label="prediction line")
    for i in range(len(x)):
        plt.vlines(x[i], y[i], y_hat[i], colors='g',
                   linestyles='dashed', alpha=0.5)
    plt.legend()
    plt.title(f"Cost: {J_value}")

    # Render
    plt.show()


if __name__ == "__main__":
    x = np.arange(1, 6)
    y = np.array([11.52434424, 10.62589482,
                 13.14755699, 18.60682298, 14.14329568])
    # Example 1:
    theta1 = np.array([18, -1])
    plot_with_loss(x, y, theta1)

    # Example 2:
    theta2 = np.array([14, 0])
    plot_with_loss(x, y, theta2)

    # Example 3:
    theta3 = np.array([12, 0.8])
    plot_with_loss(x, y, theta3)
