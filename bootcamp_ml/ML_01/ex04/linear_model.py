import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def predict_(self, x: np.ndarray) -> np.ndarray | None:
        """Computes the vector of prediction y_hat from two non-empty numpy.array.

        Args:
            x: has to be an numpy.array.
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
                or not isinstance(self.thetas, np.ndarray)
                #! or not x.ndim == 1
                or not self.thetas.shape == (2, 1)):
            return None
        #! X = x[:, np.newaxis]
        X = np.hstack((np.ones((x.shape[0], 1)), x))
        y_hat = np.dot(X, self.thetas)

        return y_hat

    def fit_(self, x, y):
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during the gradient descent
        Returns:
            new_theta: numpy.ndarray, a vector of dimension 2 * 1.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exception.
        """
        if (not isinstance(y, np.ndarray)
                or not isinstance(x, np.ndarray)
                or not y.shape == x.shape
                or not isinstance(self.thetas, np.ndarray)
                or not self.thetas.shape == (2, 1)
                or y.size == 0
                or x.size == 0
                or self.thetas.size == 0
                or not isinstance(self.alpha, float)
                or not isinstance(self.max_iter, int)):
            return None
        m = x.size

        def deriv_J0(y_hat: np.ndarray, y: np.ndarray, m):
            derivative_J0 = np.array(y_hat - y).sum() / m
            return derivative_J0

        def deriv_J1(y_hat: np.ndarray, y: np.ndarray, x: np.ndarray, m):
            values = (y_hat - y) * x
            derivative_J1 = values.sum() / m
            return derivative_J1

        step_size = self.alpha
        for i in range(self.max_iter):
            if i % 100 == 0:
                print(f'{i} / {self.max_iter}', end='\r')
            dJ0 = deriv_J0(self.predict_(x), y, m)
            dJ1 = deriv_J1(self.predict_(x), y, x, m)
            tmp_w = self.thetas[0][0] - step_size * dJ0
            tmp_b = self.thetas[1][0] - step_size * dJ1
            self.thetas = np.array([tmp_w, tmp_b]).reshape(-1, 1)
            if np.isnan(dJ0) or np.isnan(dJ1) or np.abs(dJ0) > 1e10 or np.abs(dJ1) > 1e10:
                break
            if dJ0 == 0 or dJ1 == 0:
                break
        return self.thetas

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
        J_elem = MyLinearRegression.loss_elem_(y, y_hat)
        m = y.shape[0]
        J_value = J_elem.sum() / (2 * m)
        return J_value

    def mse_(y, y_hat):
        return MyLinearRegression.loss_(y, y_hat) * 2

    def plot_evolution_cost(self, x, y, num_point=1000):
        theta0_min = self.thetas[0] - 15
        theta0_max = self.thetas[0] + 15
        theta1_min = self.thetas[1] - 5
        theta1_max = self.thetas[1] + 5
        theta1_range = np.linspace(theta1_min, theta1_max, num_point)

        # Create an array to store the cost values
        Jw = np.zeros((num_point, 6))

        # Save original theta1 to restore later
        original_theta1 = self.thetas[1].copy()

        for i, w in enumerate(theta1_range):
            for j, b in enumerate(np.linspace(theta0_min, theta0_max, 6)):
                self.thetas[1] = w
                self.thetas[0] = b
                y_hat = self.predict_(x)
                cost = MyLinearRegression.loss_(y, y_hat)
                Jw[i][j] = cost

        self.thetas[1] = original_theta1
        plt.grid(True)
        plt.ylim(bottom=Jw.min() - 5, top=(Jw.max() / 7))
        # Set x-axis limits
        plt.xlim(left=theta1_min - 2, right=theta1_max + 2)
        plt.plot(theta1_range, Jw)
        plt.xlabel('Theta1')
        plt.ylabel('Cost Function J(θ)')
        plt.title('Cost Function Evolution')
        plt.legend(['Cost Function'])
        plt.show()


MyLR = MyLinearRegression
data = pd.read_csv("../are_blue_pills_magics.csv")
Xpill = np.array(data['Micrograms']).reshape(-1, 1)
Yscore = np.array(data['Score']).reshape(-1, 1)
linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model2 = MyLR(np.array([[89.0], [-6]]))

plt.scatter(Xpill, Yscore, marker='o', label="Data points")
Y_model1 = linear_model1.predict_(Xpill)
plt.plot(Xpill, Y_model1, color='red', label="Prediction line")
plt.scatter(Xpill, Y_model1, marker='x',
            color='green', label='Predictions points')
plt.xlabel('Quantity of blue pills')
plt.ylabel('Spacecraft piloting score')
plt.legend()
plt.show()

linear_model1.plot_evolution_cost(Xpill, Yscore)


Y_model2 = linear_model2.predict_(Xpill)

print(MyLR.mse_(Yscore, Y_model1))
# 57.60304285714282
print(mean_squared_error(Yscore, Y_model1))
# 57.603042857142825
print(MyLR.mse_(Yscore, Y_model2))
# 232.16344285714285
print(mean_squared_error(Yscore, Y_model2))
# 232.16344285714285
