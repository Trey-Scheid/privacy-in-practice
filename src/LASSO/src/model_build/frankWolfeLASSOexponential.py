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

def exponential_mechanism(scores, epsilon, l):
    """
    Select vertex using exponential mechanism
    scores: utility scores for each vertex
    epsilon: privacy parameter
    """
    if not epsilon is None:
        # Normalize scores to sensitivity 1
        sensitivity = 2.0*l  # For L1 normalized data
        # prob = np.exp(-epsilon * scores / (2 * sensitivity)) # or next few lines
        
        # Normalize scores to prevent overflow
        scores = scores - np.min(scores)  # Shift scores to be non-negative
        # Compute probabilities with numerical stability
        log_prob = -epsilon * scores / (2 * sensitivity)
        log_prob = log_prob - np.max(log_prob)  # Prevent underflow
        prob = np.exp(log_prob)
    else:
        prob = scores

    prob = prob / np.sum(prob) 
    return np.random.choice(scores.shape[0], p=prob)

def frankWolfeLASSO(A, y, l=1.0, tol=1e-4, epsilon=None, delta=1e-6, K=15000, trace=True):
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
        #L1 = 2 * np.linalg.norm(A.T @ A, ord=2) # maximum columnwise: l1 norm of each column then max
        tK = int((L1**(2/3) * (n*epsilon)**(2/3)) / l**(2/3))
        K = max(min(tK, K), 1)
        t = K
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
        selected_idx = exponential_mechanism(scores, epsilon_t, l)
    
        # x_new = (1 - rho) * x_prev + rho * vertices[selected_idx, :]
        s = 1 if selected_idx < p else -1
        x_new = (1 - rho) * x_prev #+ rho * s on next line
        x_new[selected_idx % p] += s*rho
        f_new = f(x_new, A, y)
        if trace:
            convergence_criteria.append(abs(f_new - f_prev))#grad @ x_new)
        if  abs(f_new - f_prev) < tol:# or (grad @ x_new) < tol: #or np.linalg.norm(x_new - x_prev, ord=np.inf) < tol:
            t = k
            x_prev = x_new
            f_prev = f_new
            print(f"completed iteration: {t}")
            break

        x_prev = x_new
        f_prev = f_new
    
    total_budget = epsilon
    if not epsilon is None:
        total_budget = t * epsilon_t
        if total_budget < epsilon:
            print("Only spent:", total_budget)

    return {"model":x_prev, "plot":convergence_criteria, "total_budget":total_budget}
