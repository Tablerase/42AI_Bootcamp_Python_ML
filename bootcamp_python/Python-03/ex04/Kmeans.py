import sys
import numpy as np
import matplotlib.pyplot as plt
from Python_02.ex03.csvreader import CsvReader

# TODO: make fucntion to clusterize the data


class KmeansClustering:
    def __init__(self, max_iter=20, ncentroid=5):
        self.ncentroid = ncentroid  # number of centroids
        self.max_iter = max_iter  # number of max iterations to update the centroids
        self.centroids = []  # values of the centroids

    def fit(self, X):
        """
        Run the K-means clustering algorithm.
        For the location of the initial centroids, randomly pick n centroids from the dataset.

        Args:
            X: has to be an numpy.ndarray, a matrice of dimension m * n.
        Return:
            None.
        Raises:
            This function should not raise any Exception.
        """
        # ... your code ...

    def predict(self, X):
        """
        Predict from wich cluster each datapoint belongs to.

        Args:
            X: has to be an numpy.ndarray, a matrice of dimension m * n.
        Return:
            the prediction has a numpy.ndarray, a vector of dimension m * 1.
        Raises:
            This function should not raise any Exception.
        """
        # ... your code ...


if __name__ == "__main__":
    usage = f"Usage:\npython Kmeans.py filepath='../ressources/solar_system_census.csv' ncentroid=4 max_iter=30"

    try:
        inputs = sys.argv[1:]
        if not len(inputs) == 3:
            raise ValueError(f"{usage}")
        # Parsing args
        for input in inputs:
            if input.startswith('filepath='):
                filepath = input[len('filepath='):]
                continue
            if input.startswith('ncentroid='):
                ncentroid = int(input[len('ncentroid='):])
                continue
            if input.startswith('max_iter='):
                max_iter = int(input[len('max_iter='):])
                continue

        if not filepath or not ncentroid or not max_iter:
            raise ValueError(f"Invalid inputs: {inputs} \n{usage}")

        # Read dataset
        data: np.ndarray = None
        headers = None
        with CsvReader(filepath, header=True) as f:
            data = np.array(f.getdata(), dtype=float)
            height, width = data.shape
            # print(height, width)
            # print(data)
            data = data[0:, 1:]
            headers = f.getheader()

        # Fit the K-means model
        kmeans = KmeansClustering(max_iter=max_iter, ncentroid=ncentroid)
        kmeans.fit(data)

        # Predict the clusters
        predictions = kmeans.predict(data)

        # Plot the data points and centroids in 3D
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                   c=predictions, cmap='viridis', marker='o')
        # ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
        #    kmeans.centroids[:, 2], c='red', marker='x')
        ax.set_xlabel(headers[1])
        ax.set_ylabel(headers[2])
        ax.set_zlabel(headers[3])
        plt.show()

    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)

    print(f"{'Data Loading':_^60}")
