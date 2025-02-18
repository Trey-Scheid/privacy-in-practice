import numpy as np

class KMeans:
    def __init__(self, k, max_iterations=300):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
    
    # def initialize_centroids(self, data):
    #     """Randomly initialize k centroids from the data points."""
    #     n = data.shape[0]
    #     random_indices = np.random.choice(n, self.k, replace=False)
    #     self.centroids = data[random_indices]
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

