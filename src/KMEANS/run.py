import pandas as pd
import os
import src.DevicePreprocessor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import src.DPKMeans
import numpy as np
from sklearn.model_selection import train_test_split
import src.KMeans

def preprocess_data(cwd):
    """Preprocess raw device data and normalize it."""
    db_path = os.path.join(cwd, 'duckdb.db')
    pkl_path = os.path.join(cwd, 'src', 'software_data.pkl')
    #originally 10000 in paper and used this value for png
    count_parameter=10

    """following commented out code is used in real data analysis"""
    #count_parameter=10
    # for file in os.listdir(os.path.join(cwd, 'private_data', 'raw')):
    #     if file != ".DS_Store":
    #         parquet_file = os.path.join(cwd, 'private_data', 'raw', file)
    #         output_parquet_path = os.path.join(cwd, 'private_data', 'out', file)
    #         analysis = src.DevicePreprocessor.DevicePreprocessor(
    #             db_path=db_path, parquet_file=parquet_file, 
    #             pkl_path=pkl_path, output_parquet_path=output_parquet_path,
    #             count_parameter=count_parameter
    #         )
    #         analysis.preprocess()
    # data_list = []

    # for file in os.listdir(os.path.join(cwd, 'private_data', 'out')):
    #     df = pd.read_parquet(os.path.join(cwd, 'private_data', 'out', file))
    #     data_list.append(pd.DataFrame(df["l1_distance"]))
    
    # data = pd.concat(data_list, ignore_index=True)
    parquet_file=os.path.join(cwd, '..', '..','dummy_data','frgnd_v2_hist','synthetic.parquet')
    output_parquet_path = 'synthetic_out.parquet'
    analysis = src.DevicePreprocessor.DevicePreprocessor(
        db_path=db_path, parquet_file=parquet_file, 
        pkl_path=pkl_path, output_parquet_path=output_parquet_path,
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
    kmeans = src.KMeans.KMeans(k)
    kmeans.fit(train_data)
    train_loss = kmeans.compute_inertia(train_data) / train_data.shape[0]
    test_loss = kmeans.compute_inertia(test_data) / test_data.shape[0]
    return train_loss, test_loss

def compute_dpkmeans_inertia(data, k, epsilons,tau):
    """Compute DP-KMeans inertia for different epsilon values."""
    inertias = []
    for eps in epsilons:
        dp_kmeans = src.DPKMeans.DPKMeans(k=k, eps=eps,tau=tau, iterations=10)
        dp_kmeans.fit(data)
        inertias.append(dp_kmeans.compute_inertia(data) / data.shape[0])
    return inertias
def compute_dpkmeans_loss(data, k, tau_values,epsilon,test_size, random_seed):
    """Compute DP-KMeans train and test loss for different tau values."""
    train_losses = []
    test_losses = []
    
    for tau in tau_values:
        dp_kmeans = src.DPKMeans.DPKMeans(k=k, eps=epsilon, tau=tau, iterations=10)
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_seed)
        dp_kmeans.fit(train_data)
        train_loss = dp_kmeans.compute_inertia(train_data) / train_data.shape[0]
        test_loss = dp_kmeans.compute_inertia(test_data) / test_data.shape[0]
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    print(train_losses)
    print(test_losses)
    return train_losses, test_losses

def plot_results(cwd, epsilons, inertias, train_loss, test_loss, tau_values, dp_train_losses, dp_test_losses, output_path="dp_kmeans_inertia.png"):
    """Plot and save the DP-KMeans inertia along with train and test loss."""
    plt.figure(figsize=(12, 6))
    plt.plot(epsilons, inertias, marker='o', linestyle='-', color='b', label='DP-KMeans Inertia')
    plt.axhline(y=train_loss, color='g', linestyle='--', label='Train Loss '+str(round(train_loss,2)))
    plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss '+str(round(test_loss,2)))
    
    plt.xlabel("Epsilon")
    plt.ylabel("Inertia")
    plt.title("Effect of Epsilon on DP-KMeans Avg Inertia")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(cwd, '..', '..','viz','KMEANS',output_path))
    print(f"Plot saved as {output_path}")
    plt.close()

    # Plot train/test loss vs tau
    plt.figure(figsize=(12, 6))
    plt.plot(tau_values, dp_train_losses, marker='o', linestyle='-', color='g', label='DP-KMeans Train Loss')
    plt.plot(tau_values, dp_test_losses, marker='o', linestyle='-', color='r', label='DP-KMeans Test Loss')
    plt.xlabel("Tau")
    plt.ylabel("Loss")
    plt.title("Effect of Tau on DP-KMeans Train and Test Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(cwd, '..', '..', 'viz', 'KMEANS', 'dp_kmeans_tau_loss.png'))
    print("Plot saved as dp_kmeans_tau_loss.png")
    plt.close()

def main():
    cwd = os.getcwd()
    k = 3
    test_size = 0.2
    random_seed = 42
    epsilons =np.linspace(0.01, 100, 60)
    epsilon=1
    tau = 10
    np.random.seed(random_seed)
    
    # Preprocess the data
    data = preprocess_data(cwd)
    
    # Compute standard KMeans train/test loss
    train_loss, test_loss = compute_kmeans_loss(data, k, test_size, random_seed)
    
    # Compute DP-KMeans inertia for different epsilon values
    inertias = compute_dpkmeans_inertia(data, k, epsilons, tau)
    
    # Compute DP-KMeans train/test loss for different tau values
    tau_values = np.linspace(.01, 10, 30)
    dp_train_losses, dp_test_losses = compute_dpkmeans_loss(data, k, tau_values, epsilon, test_size, random_seed)
    

    plot_results(cwd, epsilons, inertias, train_loss, test_loss, tau_values, dp_train_losses, dp_test_losses, "synthetic_dp_kmeans_inertia")
if __name__ == "__main__":
    main()
