import sys
import numpy as np
import matplotlib.pyplot as plt
from Python_02.ex03.csvreader import CsvReader


class KmeansClustering:
    def __init__(self, max_iter=20, ncentroid=5):
        self.ncentroid = ncentroid  # number of centroids
        self.max_iter = max_iter  # number of max iterations to update the centroids
        self.centroids = []  # values of the centroids
        self.distances = []  # every distance for each centroid to each point

    def init_centroids(self, X: np.ndarray):
        """Randomly initialize the n centroid

        Args:
            X (np.ndarray): Data set to clusterize
        """
        items, dimensions = X.shape
        indices = np.random.choice(items, size=self.ncentroid, replace=False)
        self.centroids = X[indices]

    def points_to_nearest_cluster(self, X: np.ndarray):
        """Assign data points to nearest cluster

        Args:
            X (np.ndarray): data points
        """

    def fit(self, X):
        """
        Run the K-means clustering algorithm.
        For the location of the initial centroids, randomly pick n centroids from the dataset.

        Details:
            Hyperparameter tuning: L1 Distance
            Distances are calculated using L1 Manhattan (|x1 - x2| + |y1 - y2 | + |z1 - z2| + ...).
            Centroids are then associated with closest vectors/points
            The process is repeated max_iter
        Args:
            X: has to be an numpy.ndarray, a matrice of dimension m * n.
        Return:
            None.
        Raises:
            This function should not raise any Exception.
        """
        # Create random centroids
        if len(self.centroids) == 0:
            self.init_centroids(X)

        # Calculate distance between points and centroids
        self.distances = np.zeros((self.ncentroid, X.shape[0]))
        for i, centroid in enumerate(self.centroids):
            # print(f"Centroid {i}: {centroid}")
            self.distances[i, :] = np.sum(np.abs(centroid - X), axis=1)
            # print(self.distances[i])

    def _update_centroids(self, k_means=np.ndarray):
        """Update centroids to the mean of assigned points

        Args:
            k_means (np.ndarray): Array of points assigned to each centroid
        """
        # print(self.centroids)
        for i in range(self.ncentroid):
            if len(k_means[i]) > 0:
                self.centroids[i] = np.mean(k_means[i], axis=0)

    def _update_k_means(self, X: np.ndarray):
        """Associate data points to corresponding cluster

        Args:
            X (np.ndarray): Source data set

        Returns:
            k_means: clusters for each centroid
        """
        # Find index (array) in which the nearest point is from distances array
        k_means_indices = np.argmin(self.distances, axis=0)
        # print(k_means_indices)

        # Associate nearest centroid to point
        k_means = [[] for _ in range(self.ncentroid)]
        for i in range(self.ncentroid):
            k_means[i] = X[k_means_indices == i]
        k_means = np.array(k_means, dtype=object)

        return k_means

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

        # Repeat max_iter
        for _ in range(self.max_iter):
            # Calculate distances
            self.fit(X)
            # Associate points to clusters
            k_means = self._update_k_means(X)
            # Update Centroids
            self._update_centroids(k_means)

        return k_means


def plot_all_in_grid(predictions, headers, kmeans):
    # ___________________________ Every Cluster ___________________________

    # Plot the data points and centroids in multiple views
    # Calculate grid dimensions for subplots
    # Add 1 to account for the main plot
    grid_size = int(np.ceil(np.sqrt(ncentroid + 1)))
    fig = plt.figure(figsize=(15, 15))

    # Main 3D plot with all clusters
    ax_main = fig.add_subplot(grid_size, grid_size, 1, projection='3d')
    colors = plt.cm.rainbow(np.linspace(0, 1, ncentroid))

    # Plot all clusters in main view
    for i, cluster in enumerate(predictions):
        color = colors[i]
        ax_main.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2],
                        marker='o', color=color, alpha=0.6, label=f'Cluster {i}')
        ax_main.scatter(kmeans.centroids[i, 0], kmeans.centroids[i, 1],
                        kmeans.centroids[i, 2], marker='D', color=color,
                        s=200, edgecolor='black', label=f'Centroid {i}')

    ax_main.set_xlabel(headers[1])
    ax_main.set_ylabel(headers[2])
    ax_main.set_zlabel(headers[3])
    ax_main.set_title('All Clusters')
    ax_main.legend()

    # Individual cluster plots
    for i, cluster in enumerate(predictions):
        ax = fig.add_subplot(grid_size, grid_size, i + 2, projection='3d')
        color = colors[i]

        # Plot individual cluster
        ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2],
                   marker='o', color=color, alpha=0.6)
        ax.scatter(kmeans.centroids[i, 0], kmeans.centroids[i, 1],
                   kmeans.centroids[i, 2], marker='D', color=color,
                   s=200, edgecolor='black')

        ax.set_xlabel(headers[1])
        ax.set_ylabel(headers[2])
        ax.set_zlabel(headers[3])
        ax.set_title(f'Cluster {i}')
    plt.tight_layout()
    plt.show()


def plot_main(predictions, headers, kmeans, planets):
    # First figure with all clusters
    fig_all = plt.figure(figsize=(10, 10))
    ax_all = fig_all.add_subplot(111, projection='3d')
    colors = plt.cm.rainbow(np.linspace(0, 1, ncentroid))

    # Plot all clusters in main view
    for i, cluster in enumerate(predictions):
        color = colors[i]
        name = next(key for key, values in planets.items() if values == i)
        ax_all.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2],
                       marker='o', color=color, alpha=0.6, label=f'{name}: {len(cluster)} citizens')
        ax_all.scatter(kmeans.centroids[i, 0], kmeans.centroids[i, 1],
                       kmeans.centroids[i, 2], marker='D', color=color,
                       s=100, edgecolor='black', label=f'{name}')

    ax_all.set_xlabel(headers[1])
    ax_all.set_ylabel(headers[2])
    ax_all.set_zlabel(headers[3])
    ax_all.set_title('All Clusters')
    ax_all.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    usage = f"Usage:\npython Kmeans.py filepath='../resources/solar_system_census.csv' ncentroid=4 max_iter=30"

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
            print(f"{'entering file':_^60}")
            data = np.array(f.getdata(), dtype=float)
            height, width = data.shape
            # print(height, width)
            # print(data)
            data = data[0:, 1:]
            headers = f.getheader()

        # Fit the K-means model
        print(f"{'Clustering':_^60}")
        kmeans = KmeansClustering(max_iter=max_iter, ncentroid=ncentroid)
        kmeans.fit(data)

        # Predict the clusters
        predictions = kmeans.predict(data)

        # Color generator
        colors = plt.cm.rainbow(np.linspace(0, 1, ncentroid))

        # plot_all_in_grid(data)

        # Categorize citizens by planets
        if kmeans.ncentroid != 4:
            plot_all_in_grid(predictions, headers, kmeans)

        else:
            planets = {'Earth': None, 'Marthian Republic': None,
                       'Venus': None, 'Belt': None}
            means = np.array([np.mean(cluster, axis=0) if len(
                cluster) > 0 else np.zeros(3) for cluster in predictions])
            low_density_idx = np.argmin(means[:, 2])
            highest_height = np.argsort(means[:, 0])
            for hight in reversed(highest_height):
                if hight not in [low_density_idx]:
                    highest_height_idx = hight
                    break
            low_weight = np.argsort(means[:, 1])
            for weight in low_weight:
                if weight not in [low_density_idx, highest_height_idx]:
                    low_weight_idx = weight
                    break
            planets['Earth'] = int([i for i in range(ncentroid) if i not in [
                low_density_idx, highest_height_idx, low_weight_idx]][0])
            planets['Belt'] = low_density_idx
            planets['Marthian Republic'] = highest_height_idx
            planets['Venus'] = low_weight_idx

            # ___________________________ Main Cluster ___________________________
            plot_main(predictions, headers, kmeans, planets)

    # except NotImplementedError as e:
    #     print(e)
    #     exit(1)

    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
