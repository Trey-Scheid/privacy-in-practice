import numpy as np

class KMeans:
    def __init__(self, k, max_iterations=300):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
    
    def initialize_centroids(self, data):
        """Initialize centroids randomly."""
        n, d = data.shape
        
        # Randomly pick k unique points from the data to be the initial centroids
        indices = np.random.choice(n, self.k, replace=False)
        self.centroids = data[indices]

    def fit(self, data):
        """Perform standard k-means clustering."""
        n, d = data.shape
        
        # Step 1: Initialize centroids randomly from data points
        self.initialize_centroids(data)
        
        for iteration in range(self.max_iterations):
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
                    new_centroids[j] = np.mean(data[clusters[j]], axis=0)
            
            # Convergence check: If centroids don't change, stop early
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        
        return clusters, self.centroids
    
    def compute_inertia(self, data):
        """Compute the inertia (sum of squared distances of samples to their closest centroid)."""
        inertia = 0
        for i in range(data.shape[0]):
            distances = [np.linalg.norm(data[i] - centroid) for centroid in self.centroids]
            inertia += np.min(distances) ** 2
        return inertia

