import numpy as np

class DPKMeans:
    def __init__(self, k, eps, iterations=5):
        self.k = k
        self.iterations = iterations
        self.centroids = None
        self.epsilon = eps

    def laplace_noise(self, sensitivity, epsilon):
        """Generate Laplace noise given sensitivity and epsilon."""
        scale = sensitivity / self.epsilon
        return np.random.laplace(0, scale)

    def initialize_centroids(self, data):
        """Initialize centroids using sphere packing."""
        n, d = data.shape
        a = 1  # Start with an initial guess for the radius a
        centroids = []
        max_attempts = 1000  # Maximum number of attempts to find valid centroids
        
        # Function to check if a new centroid satisfies the conditions
        def is_valid(new_centroid, centroids, a, data_min, data_max):
            # Check if centroid is at least 'a' away from the borders
            if np.any(new_centroid <= data_min + a) or np.any(new_centroid >= data_max - a):
                return False
            # Check if centroid is at least '2a' away from other centroids
            for centroid in centroids:
                if np.linalg.norm(new_centroid - centroid) < 2 * a:
                    return False
            return True
        
        # Binary search for the largest valid radius 'a'
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        while True:
            centroids = []
            attempts = 0
            for _ in range(self.k):
                while attempts < max_attempts:
                    # Randomly generate a new candidate centroid
                    new_centroid = np.random.uniform(data_min + a, data_max - a, size=d)
                    if is_valid(new_centroid, centroids, a, data_min, data_max):
                        centroids.append(new_centroid)
                        break
                    attempts += 1
                if attempts >= max_attempts:
                    break

            # If we successfully picked all k centroids, we stop and use this radius 'a'
            if len(centroids) == self.k:
                self.centroids = np.array(centroids)
                break
            else:
                # If we failed, try a smaller radius and try again
                a /= 2  # Halve the radius

    def fit(self, data):
        """Perform standard k-means clustering with differential privacy."""
        n, d = data.shape
        t = 5
        r = 10
        sensitivity = (d * r + 1) * t

        # Step 1: Initialize centroids using sphere packing
        self.initialize_centroids(data)

        for iteration in range(self.iterations):
            # Step 2: Assign points to the nearest centroid
            clusters = [[] for _ in range(self.k)]
            for i in range(n):
                distances = [np.linalg.norm(data[i] - centroid) for centroid in self.centroids]
                assigned_cluster = np.argmin(distances)
                clusters[assigned_cluster].append(i)

            # Step 3: Update centroids
            new_centroids = np.zeros((self.k, d))
            for j in range(self.k):
                if clusters[j]:  # Avoid division by zero
                    # Compute the mean for each cluster and add Laplace noise
                    mean = np.mean(data[clusters[j]], axis=0)
                    noise = np.array([self.laplace_noise(sensitivity, self.epsilon) for _ in range(d)])
                    new_centroids[j] = mean + noise

            self.centroids = new_centroids

        return clusters, self.centroids

    def compute_inertia(self, data):
        """Compute the inertia (sum of squared distances of samples to their closest centroid)."""
        inertia = 0
        for i in range(data.shape[0]):
            distances = [np.linalg.norm(data[i] - centroid) for centroid in self.centroids]
            inertia += np.min(distances) ** 2
        return inertia
