import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f(x, A, y):
    Ax = A @ x
    diff = Ax - y
    return np.dot(diff.ravel(), diff.ravel())

def gradient(x, A, y):
    Ax = A @ x
    diff = Ax - y
    return 2 * (A.T @ diff)

def get_vertices(p):
    """Generate all vertices of L1 ball: {±1, 0, ..., 0} vectors"""
    return np.vstack([np.eye(p), -np.eye(p, dtype=int)])

def exponential_mechanism(scores, epsilon, L1):
    """
    Select vertex using exponential mechanism
    scores: utility scores for each vertex
    epsilon: privacy parameter
    """
    if not epsilon is None:
        # Normalize scores to sensitivity 1
        sensitivity = 2.0 * L1  # For L1 normalized data
        # prob = np.exp(-epsilon * scores / (2 * sensitivity)) # or next few lines
        
        # Normalize scores to prevent overflow
        scores = scores - np.min(scores)  # Shift scores to be non-negative
        # Compute probabilities with numerical stability
        log_prob = -epsilon * scores / (2 * sensitivity)
        log_prob = log_prob - np.max(log_prob)  # Prevent underflow
        prob = np.exp(log_prob)
        prob = prob / np.sum(prob) 
        return np.random.choice(scores.shape[0], p=prob)
    return np.argmax(scores)

def frankWolfeLASSOexponential(A, y, l=1.0, tol=1e-4, epsilon=None, delta=1e-6, K=15000, trace=True):
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy()
    convergence_criteria = []
    n, p = A.shape
    if y.shape[0] != n:
        raise ValueError("Dimension mismatch between A and y")
        
    if not epsilon is None:
        # Set number of iterations based on theory
        L1 = np.max(np.linalg.norm(A, ord=1, axis=0))
        tK = int((L1**(2/3) * (n*epsilon)**(2/3)) / l**(2/3))
        K = max(min(tK, K), 1)
        t = K
    else:
        L1 = None
    print(f"Total iterations: {K}")

    x_prev = A[0, :] #np.zeros(p, dtype=np.float32)
    f_prev = f(x_prev, A, y)
    
    # Generate vertices once
    vertices = get_vertices(p)
    
    # Privacy budget per iteration
    epsilon_t = epsilon / K if not epsilon is None else None
    
    for k in range(1, K+1):
        rho = 2 / (k + 2)
        grad = gradient(x_prev, A, y)
        # Compute utility scores for each vertex
        scores = vertices @ grad
        
        # Select vertex using exponential mechanism
        selected_idx = exponential_mechanism(scores, epsilon_t, L1)
    
        # x_new = (1 - rho) * x_prev + rho * vertices[selected_idx, :]
        s = 1 if selected_idx < p else -1
        x_new = (1 - rho) * x_prev #+ rho * s on next line
        x_new[selected_idx % p] += s*rho
        f_new = f(x_new, A, y)
        if trace:
            convergence_criteria.append(abs(f_new - f_prev))#grad @ x_new)
        if  abs(f_new - f_prev) < tol:# or (grad @ x_new) < tol: #or np.linalg.norm(x_new - x_prev, ord=np.inf) < tol:
            t = k
            print(f"completed iteration: {t}")
            break

        x_prev = x_new
        f_prev = f_new
    
    total_budget = epsilon
    if not epsilon is None:
        total_budget = t * epsilon_t
        if total_budget < epsilon:
            print("Only spent:", total_budget)

    return {"model":x_new, "plot":convergence_criteria, "total_budget":total_budget}

def fwOracle(grad, L1):
    # Simplified oracle computation without unnecessary allocations
    i = np.argmax(np.abs(grad))
    s = np.zeros_like(grad)
    s[i] = -np.sign(grad[i]) * L1
    return s

def frankWolfeLASSOLaplace(A, y, l=1.0, tol=0.0001, K=15000, delta=1e-6, epsilon=None, plot=False):
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy()
    n, p = A.shape
    if y.shape[0] != A.shape[0]:
            raise ValueError("Dimension mismatch between A and y")

    x_prev = A[0, :] #np.zeros(p, dtype=np.float32)  # Use float32
    f_prev = f(x_prev, A, y)
    rng = np.random.default_rng(seed=1)

    if not epsilon is None:
        L1 = np.max(np.linalg.norm(A, ord=1, axis=0))
        #better max iter for tradeoff of eps per iteration
        tK = int((n * epsilon)**(2/3) / (L1)**(2/3))
        K = max(min(tK, K), 1)
        t = K
        print(f"Total iterations: {K}")
        # Compute per-iteration privacy budget
        noise_scale = (L1 * 1 * np.sqrt(8 * K * np.log(1/delta))) / (n * epsilon) # 1 for L of vertices in l1 ball
        print(f"noise scale: {noise_scale:.3g}")
    else:
        noise_scale=0
        L1 = 1 # benign
    
    for k in range(1, K):
        rho = 2 / (2 + k)
        grad = gradient(x_prev, A, y)

        noise = rng.laplace(scale=noise_scale, size=p).reshape(p, 1)
        s = fwOracle(grad+noise, L1)
        x_new = (1 - rho) * x_prev + rho * s
        f_new = f(x_new, A, y)
        
        if  abs(f_new - f_prev) < tol:
            t = k
            break
        
        x_prev = x_new
        f_prev = f_new
    if not epsilon is None:
        total_budget = t/K * epsilon
        if total_budget < epsilon:
            print("Only spent:", total_budget)
    return x_new