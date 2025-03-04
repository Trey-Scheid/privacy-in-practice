import numpy as np
import pandas as pd

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

def exponential_mechanism(scores, epsilon, l):
    """
    Select vertex using exponential mechanism
    scores: utility scores for each vertex
    epsilon: privacy parameter
    """
    # Normalize scores to sensitivity 1
    sensitivity = 2.0*l  # For L1 normalized data
    # prob = np.exp(-epsilon * scores / (2 * sensitivity)) # or next few lines
    
    # Normalize scores to prevent overflow
    scores = scores - np.min(scores)  # Shift scores to be non-negative
    # Compute probabilities with numerical stability
    log_prob = -epsilon * scores / (2 * sensitivity)
    log_prob = log_prob - np.max(log_prob)  # Prevent underflow
    prob = np.exp(log_prob)

    prob = prob / np.sum(prob) 
    return np.random.choice(scores.shape[0], p=prob)

def frankWolfeLASSO(A, y, l=1.0, tol=1e-4, epsilon=0.1, delta=1e-6, K=15000):
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy()
        
    n, p = A.shape
    if y.shape[0] != n:
        raise ValueError("Dimension mismatch between A and y")
        
    # Set number of iterations based on theory

    L1 = np.max(np.linalg.norm(A, ord=1, axis=0))
    #L1 = 2 * np.linalg.norm(A.T @ A, ord=2) # maximum columnwise: l1 norm of each column then max
    tK = int((L1**(2/3) * (n*epsilon)**(2/3)) / l**(2/3))
    K = max(min(tK, K), 1)
    t = K
    print(f"Total iterations: {K}")

    x_prev = np.zeros(p, dtype=np.float32)
    f_prev = f(x_prev, A, y)
    
    # Generate vertices once
    vertices = get_vertices(p)
    
    # Privacy budget per iteration
    epsilon_t = epsilon / K
    
    for k in range(1, K+1):
        rho = 2 / (k + 2)
        grad = gradient(x_prev, A, y)
        # Compute utility scores for each vertex
        scores = np.dot(vertices, grad)
        
        # Select vertex using exponential mechanism
        selected_idx = exponential_mechanism(scores, epsilon_t, l)
        s = vertices[selected_idx, :]
        
        x_new = (1 - rho) * x_prev + rho * s
        f_new = f(x_new, A, y)

        if  abs(f_new - f_prev) < tol:
            t = k
            break

        x_prev = x_new
        f_prev = f_new
    
    total_budget = t * epsilon_t
    if total_budget < epsilon:
        print("Only spent:", total_budget)
    return x_prev#, total_budget
