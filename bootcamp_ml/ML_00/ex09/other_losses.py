import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


def mse_(y, y_hat):
    """
    Description:
        Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be a numpy.array, a two-dimensional vector of shape m * 1.
    Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if (not isinstance(y, np.ndarray)
            or not y.ndim == 2
            or not isinstance(y_hat, np.ndarray)
            or not y_hat.ndim == 2
            or not y.shape == y_hat.shape):
        return None
    mse_elem = (y_hat - y)**2
    MSE_value: float = mse_elem.sum() / (y.size)
    return MSE_value


def rmse_(y, y_hat):
    """
    Description:
        Calculate the RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    MSE = mse_(y, y_hat)
    if MSE == None:
        return None
    return sqrt(MSE)


def mae_(y, y_hat):
    """
    Description:
    Calculate the MAE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a two-dimensional array of shape m * 1.
    y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
    Returns:
    mae: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    mae_elem = np.abs(y_hat - y)
    mae_value: float = mae_elem.sum() / (y.size)
    return mae_value


def r2score_(y, y_hat):
    """
    Description:
    Calculate the R2score between the predicted output and the output.

    R2 = 1 - RSS / TSS
        RSS : sum of squares of residuals
        TSS : total sum of squares
    Args:
    y: has to be a numpy.array, a two-dimensional array of shape m * 1.
    y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
    Returns:
    r2score: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """

    RSS = np.array((y_hat - y)**2).sum()
    TSS = np.array((y_hat - y.mean())**2).sum()
    R2 = 1 - RSS / TSS
    return R2


if __name__ == "__main__":
    # Example 1:
    x = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    # Mean-squared-error
    # your implementation
    res = mse_(x, y)
    print("MSE:", res)
    # Output:
    # 4.285714285714286
    # sklearn implementation
    src = mean_squared_error(x, y)
    print("Real MSE:", src)
    # Output:
    # 4.285714285714286

    # Root mean-squared-error
    # your implementation
    res = rmse_(x, y)
    print("RMSE:", res)
    # Output:
    # 2.0701966780270626
    # sklearn implementation not available: take the square root of MSE
    src = sqrt(mean_squared_error(x, y))
    print("Real RMSE:", src)
    # Output:
    # 2.0701966780270626

    # Mean absolute error
    # your implementation
    res = mae_(x, y)
    print("MAE:", res)
    # Output:
    # 1.7142857142857142
    # sklearn implementation
    src = mean_absolute_error(x, y)
    print("Real MAE:", src)
    # Output:
    # 1.7142857142857142

    # R2-score
    # your implementation
    res = r2score_(x, y)
    print("R2:", res)
    # Output:
    # 0.9681721733858745
    # sklearn implementation
    src = r2_score(x, y)
    print("Real R2:", src)
    # Output:
    # 0.9681721733858745
