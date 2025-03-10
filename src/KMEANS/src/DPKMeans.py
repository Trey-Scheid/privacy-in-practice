import numpy as np

class DPKMeans:
    def __init__(self, k, eps,tau, iterations=5):
        self.k = k
        self.iterations = iterations
        self.centroids = None
        self.epsilon = eps
        self.epsilon_iter=eps/iterations
        self.tau=tau

    def laplace_noise(self, sensitivity,epsilon):
        """Generate Laplace noise given sensitivity and epsilon."""
        scale = sensitivity / epsilon
        return np.random.laplace(0, scale)

    def initialize_centroids(self, data):
        """Initialize centroids randomly."""
        n, d = data.shape
        
        # Randomly pick k unique points from the data to be the initial centroids
        indices = np.random.choice(n, self.k, replace=False)
        self.centroids = np.clip(data[indices],-self.tau,self.tau)


    def fit(self, data):
        """Perform standard k-means clustering with differential privacy."""
        n, d = data.shape

        # Step 1: Initialize centroids using sphere packing
        self.initialize_centroids(data)

        for iteration in range(self.iterations):
            # Step 2: Assign points to the nearest centroid
            clusters = [[] for _ in range(self.k)]
            for i in range(n):
                distances = [np.linalg.norm(np.clip(data[i],-self.tau,self.tau) - centroid) for centroid in self.centroids]
                assigned_cluster = np.argmin(distances)
                clusters[assigned_cluster].append(i)

            # Step 3: Update centroids
            new_centroids = np.zeros((self.k, d))
            for j in range(self.k):
                if clusters[j]:  # Avoid division by zero
                    # Compute the mean for each cluster and add Laplace noise
                    clipped_data = np.clip(data[clusters[j]], -self.tau, self.tau)
                    noisy_sum = np.sum(clipped_data, axis=0)+self.laplace_noise(self.tau*2,self.epsilon_iter)
                    noisy_count=max(1,clipped_data.shape[0]+self.laplace_noise(1,self.epsilon_iter))
                    new_centroids[j] = noisy_sum/noisy_count

            self.centroids = new_centroids

        return clusters, self.centroids

    def compute_inertia(self, data):
        """Compute the inertia (sum of squared distances of samples to their closest centroid)."""
        inertia = 0
        for i in range(data.shape[0]):
            distances = [np.linalg.norm(data[i] - centroid) for centroid in self.centroids]
            inertia += np.min(distances) ** 2
        return inertia
