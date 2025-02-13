import numpy as np
import pandas as pd

def f(x, A, y):
    # Compute in chunks to avoid large intermediate matrices
    Ax = A @ x
    diff = Ax - y
    return np.dot(diff.ravel(), diff.ravel())

def gradient(x, A, y):
    # Compute gradient in steps to reduce memory footprint
    Ax = A @ x
    diff = Ax - y
    return 2 * (A.T @ diff)

def fwOracle(grad, l, noise_scale=0):
    # Simplified oracle computation without unnecessary allocations
    i = np.argmax(np.abs(grad + 
        np.random.laplace(scale=noise_scale, size=grad.shape[0])
        .reshape(grad.shape[0],1))
        )
    s = np.zeros_like(grad)
    s[i] = -np.sign(grad[i]) * l
    return s

def frankWolfeLASSO(A, y, l=500, tol=0.0001, K=15000, delta=1e-6, epsilon=None):
    if type(A) == pd.DataFrame:
         A = A.to_numpy()
    if type(y) == pd.DataFrame | type(y) == pd.Series:
         y = y.to_numpy()
    p = A.shape[1]
    if y.shape[0] != A.shape[0]:
            raise ValueError("Dimension mismatch between A and y")

    x_prev = np.zeros((p, 1), dtype=np.float32)  # Use float32
    y = y.reshape(-1, 1)
    f_prev = f(x_prev, A, y)

    if not epsilon is None:
        L1 = 2 * np.linalg.norm(A.T @ A, ord=2)
        # Compute per-iteration privacy budget
        noise_scale = (L1 * l * np.sqrt(8 * K * np.log(1/delta))) / (A.shape[0] * epsilon)
        print(f"noise scale: {noise_scale:.3g}")
    else:
        noise_scale=0
    
    # Circular buffer for history
    #history_size = min(100, K)  # Store limited history
    #convergence_history = np.zeros(history_size, dtype=np.float32)
    #hist_idx = 0
    
    for k in range(1, K):
        rho = 2 / (2 + k)
        grad = gradient(x_prev, A, y)
        s = fwOracle(grad, l, noise_scale=noise_scale)
        x_new = (1 - rho) * x_prev + rho * s
        f_new = f(x_new, A, y)
        
        # Store history in circular buffer
        #convergence_history[hist_idx] = f_new
        #hist_idx = (hist_idx + 1) % history_size
        
        if (epsilon is None) and (abs(f_new - f_prev) < tol):
            break
            
        x_prev = x_new
        f_prev = f_new
    
    return x_prev.flatten()#, convergence_history
