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

def fwOracle(grad, l):
    # Simplified oracle computation without unnecessary allocations
    i = np.argmax(np.abs(grad))
    s = np.zeros_like(grad)
    s[i] = -np.sign(grad[i]) * l
    return s

def frankWolfeLASSO(A, y, l=500, tol=0.0001, K=15000, delta=1e-6, epsilon=None):
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy()
    n, p = A.shape
    if y.shape[0] != A.shape[0]:
            raise ValueError("Dimension mismatch between A and y")

    x_prev = np.zeros((p, 1), dtype=np.float32)  # Use float32
    y = y.reshape(-1, 1)
    f_prev = f(x_prev, A, y)
    rng = np.random.default_rng(seed=1)

    if not epsilon is None:
        L1 = 2 * np.linalg.norm(A.T @ A, ord=2)
        tK = int((n * epsilon)**(2/3) / (L1 * l)**(2/3))
        K = max(min(tK, K), 1)
        t = K
        print(f"Total iterations: {K}")
        # Compute per-iteration privacy budget
        noise_scale = (L1 * l * np.sqrt(8 * K * np.log(1/delta))) / (n * epsilon)
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

        noise = rng.laplace(scale=noise_scale, size=p).reshape(p, 1)
        s = fwOracle(grad+noise, l)
        x_new = (1 - rho) * x_prev + rho * s
        f_new = f(x_new, A, y)
        
        # Store history in circular buffer
        #convergence_history[hist_idx] = f_new
        #hist_idx = (hist_idx + 1) % history_size
        
        if  abs(f_new - f_prev) < tol:
            t = k
            break
        
        x_prev = x_new
        f_prev = f_new
    if not epsilon is None:
        total_budget = t/K * epsilon
        if total_budget < epsilon:
            print("Only spent:", total_budget)
    return x_prev.flatten()#, convergence_history
