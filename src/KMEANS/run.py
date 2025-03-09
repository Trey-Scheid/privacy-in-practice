import pandas as pd
import os
import src.KMEANS.src.DevicePreprocessor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import src.KMEANS.src.DPKMeans
import numpy as np
from sklearn.model_selection import train_test_split
import src.KMEANS.src.KMeans
import json
def plot_1d_clusters(data, cluster_centers, output_path="1d_clusters.png"):
    """Plot data points along a 1D line and mark cluster centers with upside-down triangles."""
    plt.figure(figsize=(12, 10))  # Wider aspect ratio
    
    # Plot data points as small gray dots
    plt.scatter(data, np.zeros_like(data), color="#31363F", alpha=0.5, label="Data Points", s=10)
    
    # Plot cluster centers as upside-down triangles (v) at y=0.05
    for center in cluster_centers:
        plt.scatter(center, 0, color="#76ABAE", marker='o', s=200)

    # Remove y-axis labels and ticks
    plt.yticks([])

    # Remove grid lines
    plt.grid(False)

    # Remove bounding box
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)

    plt.xlabel("Normalized L1 Distance (Scaled to [-10, 10])")
    plt.title("1D Clustering Visualization")
    plt.legend(["Data Points", "Cluster Centers"])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    

def preprocess_data(cwd,parquet_path,output_parquet_path):
    """Preprocess raw device data and normalize it."""
    #originally 10000 in paper and used this value for png
    count_parameter=10
    parquet_path=os.path.join(cwd,parquet_path)
    output_parquet_path = 'synthetic_out.parquet'
    analysis = src.KMEANS.src.DevicePreprocessor.DevicePreprocessor(
        parquet_file=parquet_path, 
        output_parquet_path=output_parquet_path,
        count_parameter=count_parameter
    )
    analysis.preprocess()
    df= pd.read_parquet(output_parquet_path)
    data=pd.DataFrame(df["l1_distance"])
    os.remove(output_parquet_path)
    scaler = MinMaxScaler(feature_range=(-10, 10))
    data=scaler.fit_transform(data)
    return data

def compute_kmeans_loss(data, k, test_size, random_seed):
    """Compute train and test loss using standard KMeans."""
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_seed)
    kmeans = src.KMEANS.src.KMeans.KMeans(k)
    kmeans.fit(train_data)
    train_loss = kmeans.compute_inertia(train_data) / train_data.shape[0]
    test_loss = kmeans.compute_inertia(test_data) / test_data.shape[0]
    return train_loss, test_loss

def compute_dpkmeans_inertia(data, k, epsilons, tau):
    """Compute DP-KMeans inertia for different epsilon values."""
    inertias = []
    for eps in epsilons:
        dp_kmeans = src.KMEANS.src.DPKMeans.DPKMeans(k=k, eps=eps, tau=tau, iterations=10)
        dp_kmeans.fit(data)  # Train on training set
        inertia = dp_kmeans.compute_inertia(data) / data.shape[0]  # Evaluate on test set
        inertias.append(inertia)
    
    # Normalize inertia values between 0 and 1
    inertias = [1-((x) / (max(inertias))) for x in inertias]

    return inertias

def results(cwd, epsilons, inertias, train_loss, test_loss):
    """Return DP-KMeans inertia"""
    df = pd.DataFrame({
    'epsilon': epsilons,
    'utility':inertias,
    'task': ['KMEANS'] * len(epsilons)})
    return df

def main(**params):
    kmeans_params = params.get("kmeans-params")
    k = kmeans_params.get('k') 
    epsilon = kmeans_params.get('epsilon')  
    tau = kmeans_params.get('tau') 
    parquet_path=kmeans_params.get('parquet_path')
    output_parquet_path=kmeans_params.get('output_parquet_path')
    output_cluster_path=kmeans_params.get('output_cluster_path')

    cwd = os.getcwd()
    test_size = 0.2
    random_seed = 42
    epsilons = np.logspace(-2, 2, 100)
    np.random.seed(random_seed)
    
    # Preprocess the data
    data = preprocess_data(cwd,parquet_path,output_parquet_path)
    
    # Compute standard KMeans train/test loss
    train_loss, test_loss = compute_kmeans_loss(data, k, test_size, random_seed)
    
    # Compute DP-KMeans inertia for different epsilon values
    inertias = compute_dpkmeans_inertia(data, k, epsilons, tau)
    mod=src.KMEANS.src.DPKMeans.DPKMeans(k=k, eps=epsilon, tau=tau, iterations=10)
    mod.fit(data)
    cluster_centers = mod.centroids.flatten()
    # Plot results
    plot_1d_clusters(data, cluster_centers, os.path.join(cwd, output_cluster_path))
    return results(cwd, epsilons, inertias, train_loss, test_loss)
    

if __name__ == "__main__":
    with open("config/run.json") as fh:
        params = json.load(fh)
        main(**params)
