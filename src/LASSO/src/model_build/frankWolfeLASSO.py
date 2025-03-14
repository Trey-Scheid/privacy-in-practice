"""
File: frankWolfeLASSO.py
Author: Trey Scheid
Date: last modified 03/2025
Description: 3 Frank-Wolfe model solvers for lasso regression: two private implementations (supposed to be equivalent but in practice are not), and a traditional, used by train.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from autodp.calibrator_zoo import ana_gaussian_calibrator
from autodp.mechanism_zoo import ExactGaussianMechanism
# from autodp.transformer_zoo import ComposeGaussian

##### HELPER FUNCTIONS #####

def f(x, A, y):
    """
    Loss function for lasso regression

    Args:
        x (array-like): weight vector to evaluate loss function at
        A (np.ndarray of shape (n_samples, n_features)): Data
        y (array-like): correct outcomes

    Returns:
        float: mean square error
    """         
    Ax = A @ x
    diff = Ax - y
    return np.dot(diff.ravel(), diff.ravel()) / A.shape[0]

def gradient(x, A, y):
    """
    Gradient wrt x of the Loss function of Lasso Regression

    Args:
        x (array-like): weight vector to evaluate loss function at
        A (np.ndarray of shape (n_samples, n_features)): Data
        y (array-like): correct outcomes

    Returns:
        nd.array: gradient
    """
    Ax = A @ x
    diff = Ax - y
    return 2 * (A.T @ diff) / A.shape[0]

def check_l_ball(x, l, log=False):
    """
    Check if vector x is within the L1 ball of radius l.
    
    Args:
        x (numpy.ndarray): The vector to check
        l (float): The radius of the L1 ball
    
    Returns:
        bool: True if x is within the L1 ball, False otherwise
    """
    l1_norm = np.linalg.norm(x, ord=1)
    if log:
        if l1_norm > l:
            print(f"x is outside the L1 ball radius {l}")
    return l1_norm <= l

def find_L1(X, l):
    """
    Computes Estimate of the l1-lipschitz of lasso loss

    Args:
        X (array-like): 2D data
        l (float): regularization strength, constrain set size

    Returns:
        L1: estimated lipschitz constant
    """
    n = X.shape[0]
    X_spectral_norm = np.linalg.svd(X, compute_uv=False)[0]
    L = (4 * l / n) * X_spectral_norm**2
    return L

###### METHOD ONE #######

def ExponentialMechanism(A, y, l=1.0, tol=1e-4, epsilon=None, delta=1e-6, K=15000, trace=True, normalize=True, clip_sd=np.inf):
    """
    Private Frank-Wolfe Lasso Regression model using the exponential mechanism for vertex selection.

    Args:
        A (array-like): 2D Data
        y (array-like): outcomes
        l (float, optional): regularization by constrianing solution to l1-ball with radius l. Defaults to 1.0.
        tol (float, optional): Desired optimality guarantee instead of K. Defaults to 1e-4.
        epsilon (float, optional): Privacy budget for model training (instead of K). Defaults to None.
        delta (float, optional): Privacy bias term. Defaults to 1e-6.
        K (int, optional): maximum number of iterations unless budget used first. Defaults to 15000.
        trace (bool, optional): save training information to ExponentialMechanism.plot. Defaults to True.
        normalize (bool, optional): Use 20% budget to normalize data (necessary for noise size). Defaults to True.
        clip_sd (float, optional): value to cut off outliers for noise scaling efficacy. Defaults to np.inf.

    Returns:
        output["model"]: coefficients (no intercept added)
        output["plot"]: values for each iteration
        output["total_budget"]: privacy budget used
    """
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    elif isinstance(A, list):
        A = np.array(A)
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy()
    elif isinstance(y, list):
        y = np.array(y)
    
    convergence_criteria = []
    n, p = A.shape
    if y.shape[0] != n:
        raise ValueError("Dimension mismatch between A and y")
    
    Ay = np.hstack((A, y.reshape(len(y), 1)))

    # non-private clipping
    if not clip_sd is None:
        Ay = np.clip(Ay, -clip_sd, clip_sd)
        # Ay, clip_maxes = clip_normalize_data(Ay, sd=clip_sd) # use None to skip
        # A, y = Ay[:,:-1], Ay[:, -1]
        # assert check_normalization(Ay)
    
    if normalize and not epsilon is None:
        Ay, used_budget, maxes = private_normalize_data(Ay, epsilon*.2, delta=delta)
        assert check_normalization(Ay)
        A, y = Ay[:,:-1], Ay[:, -1]
        epsilon -= used_budget
        maxes, scale_y = maxes[:-1], maxes[-1] # don't need the scale for y
    elif normalize:
        Ay, _, maxes = private_normalize_data(Ay, epsilon=None, delta=delta)
        assert check_normalization(Ay)
        A, y = Ay[:,:-1], Ay[:, -1]
        Ay = np.hstack((A, y.reshape(len(y), 1)))
        maxes, scale_y = maxes[:-1], maxes[-1]
    else:
        maxes = np.ones(Ay.shape[1])

    if not epsilon is None:
        # Set number of iterations based on theory
        # L1 = np.max(np.linalg.norm(Ay, ord=1, axis=0))
        L1 = find_L1(A, l)
        # print("L1:", L1)
        tK = int((L1**(2/3) * (n*epsilon)**(2/3)) / l**(2/3))
        if tK < K: print(tK, "iterations is more optimal")
        K = max(min(tK, K), 1)
        t = K
    else:
        L1 = None
    # print(f"Total iterations: {K}")

    x_prev = np.zeros(p, dtype=np.float32)
    f_prev = f(x_prev, A, y)
    
    # Generate vertices once
    vertices = get_vertices(p) * l
    
    # Privacy budget per iteration
    # epsilon_t = epsilon / K if not epsilon is None else None
    
    for k in range(1, K+1):
        rho = 2 / (k + 2)
        grad = gradient(x_prev, A, y)
        # Compute utility scores for each vertex
        scores = vertices @ grad
        
        # Select vertex using exponential mechanism
        selected_idx = exponential_mechanism(scores, epsilon, l, L1, n)
    
        # x_new = (1 - rho) * x_prev + rho * vertices[selected_idx, :]
        s = l if selected_idx < p else -l
        x_new = (1 - rho) * x_prev #+ rho * s on next line
        x_new[selected_idx % p] += s*rho
        f_new = f(x_new, A, y)
        check_l_ball(x_new, l, log=True)
        if trace:
            convergence_criteria.append(f_new)#grad @ x_new)
        # if (k > 1) and (abs(f_new - f_prev) < tol):# or (grad @ x_new) < tol: #or np.linalg.norm(x_new - x_prev, ord=np.inf) < tol:
        #     t = k
        #     print(f"converged at: {t}")
        #     break

        x_prev = x_new
        f_prev = f_new

    total_budget = epsilon
    if not epsilon is None:
        total_budget = t * epsilon / K
        if total_budget < epsilon:
            print("Only spent:", total_budget)
    if normalize:
        # trained on small data so coef are too large, resizing
        x_new = np.where(maxes != 0, x_new / (maxes+1e-10), x_new)
        #if not epsilon is None:
        x_new *= scale_y
    # if not clip_sd is None:
    #     x_new = np.where(clip_maxes[:-1] != 0, x_new / (clip_maxes[:-1] + 1e-10), x_new)
    #     x_new *= clip_maxes[-1]

    return {"model":x_new, "plot":convergence_criteria, "total_budget":total_budget}

def get_vertices(p):
    """
    Generate all vertices of L1 ball: {±1, 0, ..., 0} vectors

    Args:
        p (int): dimension of constraint set, dimension of data

    Returns:
        array-like: 2D array, each row is a vertex +-e_i
    """
    return np.vstack([np.eye(p), -np.eye(p, dtype=int)])

def exponential_mechanism(scores, epsilon, l, L1, n):
    """
    select index of vertex with exponential mechanism

    Args:
        scores (array-like): 2D array stores utility at each vertex
        epsilon (float): privacy parameter for this iteration
        l (float): size of constraint
        L1 (float): l1-lipschitz for Loss
        n (int): size of data for sensitivity

    Returns:
        int: index of scores which minimizes inner product of gradient and s
    """
    if not epsilon is None:
        # Normalize scores to sensitivity 1
        sensitivity = l * L1 / n # For L1 normalized data
        # prob = np.exp(-epsilon * scores / (2 * sensitivity)) # next few lines
        
        # Normalize scores to prevent overflow
        scores = scores - np.max(scores)  # Shift scores to be non-negative
        # Compute probabilities with numerical stability
        log_prob = -epsilon * scores / (2 * sensitivity)
        log_prob = log_prob - np.max(log_prob)  # Prevent underflow
        prob = np.exp(log_prob) # exponential mech
        prob = prob / np.sum(prob)
        # maximal p at negative of utility function
        return np.random.choice(scores.shape[0], p=prob)
    # non-private, always choose the best
    return np.argmin(scores)


##### METHOD TWO #####

def fwOracle(grad, l):
    """
    Oracle computes vertex that minimizes the dot product of gradient and s

    Args:
        grad (array-like): gradient of loss at current solution
        l (float): constrain size

    Returns:
        array-like: update direction
    """
    # Simplified oracle computation without unnecessary allocations
    i = np.argmax(np.abs(grad))
    s = np.zeros_like(grad)
    s[i] = -np.sign(grad[i]) * l
    return s

def LaplaceNoise(A, y, l=1.0, tol=0.0001, K=15000, delta=1e-6, epsilon=None, trace=True, normalize=True, clip_sd=np.inf):
    """
    Private Frank-Wolfe Lasso Regression model adding Laplace noise to gradient before oracle selection

    Args:
        A (array-like): 2D Data
        y (array-like): outcomes
        l (float, optional): regularization by constrianing solution to l1-ball with radius l. Defaults to 1.0.
        tol (float, optional): Desired optimality guarantee instead of K. Defaults to 1e-4.
        epsilon (float, optional): Privacy budget for model training (instead of K). Defaults to None.
        delta (float, optional): Privacy bias term. Defaults to 1e-6.
        K (int, optional): maximum number of iterations unless budget used first. Defaults to 15000.
        trace (bool, optional): save training information to ExponentialMechanism.plot. Defaults to True.
        normalize (bool, optional): Use 20% budget to normalize data (necessary for noise size). Defaults to True.
        clip_sd (float, optional): value to cut off outliers for noise scaling efficacy. Defaults to np.inf.

    Returns:
        output["model"]: coefficients (no intercept added)
        output["plot"]: values for each iteration
        output["total_budget"]: privacy budget used
    """
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy()
    convergence_criteria = []
    n, p = A.shape
    if y.shape[0] != n:
        raise ValueError("Dimension mismatch between A and y")
    
    Ay = np.hstack((A, y.reshape(len(y), 1)))

    # non-private clipping
    if not clip_sd is None:
        Ay = np.clip(Ay, -clip_sd, clip_sd)
        # Ay, clip_maxes = clip_normalize_data(Ay, sd=clip_sd) # use None to skip
        # A, y = Ay[:,:-1], Ay[:, -1]
        # assert check_normalization(Ay)

    if normalize and not epsilon is None:
        Ay, used_budget, maxes = private_normalize_data(Ay, epsilon*.2, delta=delta)
        assert check_normalization(Ay)
        A, y = Ay[:,:-1], Ay[:, -1]
        epsilon -= used_budget
        maxes, scale_y = maxes[:-1], maxes[-1] # don't need the scale for y
    elif normalize:
        Ay, _, maxes = private_normalize_data(Ay, epsilon=None, delta=delta)
        assert check_normalization(Ay)
        A, y = Ay[:,:-1], Ay[:, -1]
        Ay = np.hstack((A, y.reshape(len(y), 1)))
        maxes, scale_y = maxes[:-1], maxes[-1]
    else:
        maxes = np.ones(Ay.shape[1])

    x_prev = np.zeros(p, dtype=np.float32)
    f_prev = f(x_prev, A, y)
    rng = np.random.default_rng(seed=1)

    if not epsilon is None:
        # L1 = np.max(np.linalg.norm(Ay, ord=1, axis=0))
        L1 = find_L1(A, l)
        # print("L1:", np.round(L1), np.round(find_L1(A, l)))
        #better max iter for tradeoff of eps per iteration
        tK = int((L1**(2/3) * (n*epsilon)**(2/3)) / l**(2/3))
        K = max(min(tK, K), 1)
        t = K
        print(f"Total iterations: {K}")
        # Compute per-iteration privacy budget
        noise_scale = (L1 * l * np.sqrt(8 * K * np.log(1/delta))) / (n * epsilon) # 1 for L of vertices in l1-ball
        print(f"noise scale: {noise_scale:.3g}")
    else:
        noise_scale=0
        L1 = 1 # benign
    
    for k in range(1, K):
        rho = 2 / (2 + k)
        grad = gradient(x_prev, A, y)

        noise = rng.laplace(scale=noise_scale, size=p)#.reshape(p, 1)
        s = fwOracle(grad+noise, l)
        x_new = (1 - rho) * x_prev + rho * s
        f_new = f(x_new, A, y)
        check_l_ball(x_new, l, log=True)
        if trace:
            convergence_criteria.append(f_new)#grad @ x_new)
        # if  (k > 1) and (abs(f_new - f_prev) < tol):# or (grad @ x_new) < tol: #or np.linalg.norm(x_new - x_prev, ord=np.inf) < tol:
        #     t = k
        #     break
        
        x_prev = x_new
        f_prev = f_new

    # check_l_ball(x_new, l, log=True)

    total_budget = epsilon
    if not epsilon is None:
        total_budget = t/K * epsilon
        if total_budget < epsilon:
            print("Only spent:", total_budget)
    if normalize:
        # trained on small data so coef are too large, resizing
        x_new = np.where(maxes != 0, x_new / (maxes+1e-10), x_new)
        #if not epsilon is None:
        x_new *= scale_y
    # if not clip_sd is None:
    #     x_new = np.where(clip_maxes[:-1] != 0, x_new / (clip_maxes[:-1] + 1e-10), x_new)
    #     x_new *= clip_maxes[-1]

    return {"model":x_new, "plot":convergence_criteria, "total_budget":total_budget}

#### CLIPPING AND NORMALIZING ####

def private_normalize_data(data, epsilon=None, delta=1e-6):
    """
    use privacy budget to normalize the data -1, 1

    Args:
        data (array-like): data
        epsilon (float, optional): privacy budget for this task. Defaults to None.
        delta (float, optional): privacy parameter. Defaults to 1e-6.

    Returns:
        array-like: normalized data
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    
    n, d = data.shape
    
    # Calculate sensitivity
    # sensitivity = 2  # For l-infinity norm
    
    if not epsilon is None:
        epsilon_prime = epsilon / d

        # Initialize calibrator
        calibrator = ana_gaussian_calibrator()

        mechanism = calibrator.calibrate(
        mech_class=ExactGaussianMechanism,
        eps=epsilon_prime,
        delta=delta / d
        )
        sigma = mechanism.params['sigma']

        # Find noisy max for each column
        col_maxes = np.quantile(np.abs(data), 0.95, axis=0)
    else:
        sigma=0
        col_maxes = np.max(np.abs(data), axis=0)
    rng = np.random.default_rng(seed=2)
    noisy_maxes = col_maxes + rng.normal(loc=0, scale=sigma, size=d)
    
    # Normalize using noisy max
    normalized_data = data / np.maximum(1, noisy_maxes) #don't enlarge any data
    
    # Clip to [-1, 1]
    normalized_data = np.clip(normalized_data, -1, 1)
    
    # Calculate total privacy cost using advanced composition?
    used_epsilon = epsilon
    
    return normalized_data, used_epsilon, col_maxes

def clip_normalize_data(data, sd=None):
    """
    clip data to sd, then normalize (all non-private). if sd == None just normalizes

    Args:
        data (array-like): data
        sd (float, optional): abs clipping value. Defaults to None.

    Returns:
        normalized_data: normalized data
        col_maxes: columnwise maximums for rescaling data
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    
    n, d = data.shape
    
    if (not sd is None) and (not sd == np.inf):
        data = np.clip(data, -sd, sd)

    col_maxes = np.max(np.abs(data), axis=0)
    
    normalized_data = np.where(col_maxes != 0, data / col_maxes, data)
    
    # Clip to [-1, 1]
    normalized_data = np.clip(normalized_data, -1, 1)
    
    return normalized_data, col_maxes

def check_normalization(X):
    """
    check X is normal, no abs(values) > 1

    Args:
        X (array-like): data

    Returns:
        bool: whether X is normal
    """
    if np.any(np.abs(X) > 1):
        print("Warning: Some values have absolute value greater than 1")
        return False
    return True

#### FOR NON PRIVATE ####

def FW_NonPrivate(A, y, l=1.0, K=15000, tol=1e-4, trace=False, normalize=False, clip_sd=None, epsilon=None, delta=None):
    """
    Frank-Wolfe Lasso Regression model.

    Args:
        A (array-like): data
        y (array-like): outcomes
        l (float, optional): regularization parameter, constrain size. Defaults to 1.0.
        K (int, optional): maximum iterations. Defaults to 15000.
        tol (float, optional): optional stopping condition. Defaults to 1e-4.
        trace (bool, optional): save plot values. Defaults to False.
        normalize (bool, optional): normalize the input before training. Defaults to False.
        clip_sd (float, optional): clip data before processing. Defaults to None.
        epsilon (float, optional): for train compatibility, not used. Defaults to None.
        delta (float, optional): for train compatibility, not used. Defaults to None.

    Raises:
        ValueError: Dimension of A and y must match

    Returns:
        output["model"]: coefficients (no intercept added)
        output["plot"]: values for each iteration
        output["total_budget"]: privacy budget used
    """
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy()
    convergence_criteria = []
    n, p = A.shape
    if y.shape[0] != n:
        raise ValueError("Dimension mismatch between A and y")
    
    Ay = np.hstack((A, y.reshape(len(y), 1)))

    # non-private clipping
    if not clip_sd is None:
        Ay, clip_maxes = clip_normalize_data(Ay, sd=clip_sd) # use None to skip
        A, y = Ay[:,:-1], Ay[:, -1]
        assert check_normalization(Ay)
    
    if normalize:
        Ay, maxes = private_normalize_data(Ay)
        assert check_normalization(Ay)
        A, y = Ay[:,:-1], Ay[:, -1]
        Ay = np.hstack((A, y.reshape(len(y), 1)))
        maxes, scale_y = maxes[:-1], maxes[-1]
    else:
        maxes = np.ones(Ay.shape[1])

    x_prev = np.zeros(p, dtype=np.float32)
    f_prev = f(x_prev, A, y)
    
    # Generate vertices once
    vertices = get_vertices(p) * l
    
    # Privacy budget per iteration    
    for k in range(1, K+1):
        rho = 2 / (k + 2)
        grad = gradient(x_prev, A, y)
        # Compute utility scores for each vertex
        scores = vertices @ grad
        selected_idx = np.argmax(scores)

        # could also 
        selected_idx = np.argmax(np.abs(grad))
        s = -np.sign(grad[selected_idx]) * l

        # s = l if selected_idx < p else -l
        # if (grad != 0).sum() == 0:
        #     s = 0
        x_new = (1 - rho) * x_prev #+ rho * s on next line
        x_new[selected_idx % p] += s*rho
        f_new = f(x_new, A, y)
        check_l_ball(x_new, l, log=True)
        if trace:
            convergence_criteria.append(f_new)
        if (k > 1) and (abs(f_new - f_prev) < tol):
            t = k
            print(f"converged at: {t}")
            break

        x_prev = x_new
        f_prev = f_new

    if normalize:
        # trained on small data so coef are too large, resizing
        x_new = np.where(maxes != 0, x_new / (maxes+1e-10), x_new)
        #if not epsilon is None:
        x_new *= scale_y
    if not clip_sd is None:
        x_new = np.where(clip_maxes[:-1] != 0, x_new / (clip_maxes[:-1] + 1e-10), x_new)
        x_new *= clip_maxes[-1]

    return {"model":x_new, "plot":convergence_criteria}

