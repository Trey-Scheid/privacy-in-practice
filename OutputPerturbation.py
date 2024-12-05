from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from autodp.autodp_core import Mechanism
from autodp.mechanism_zoo import GaussianMechanism
from autodp.calibrator_zoo import eps_delta_calibrator
from autodp.mechanism_zoo import ExactGaussianMechanism, PureDP_Mechanism



#import data
df = pd.read_csv('./data/dataset.csv')

#convert categorical column to multiple binary
df=pd.get_dummies(df,columns=["channel"])

dataset = df

# Let's extract the relevant information from the sklearn dataset object
X = dataset[[x for x in dataset.columns if x!="anomaly"]]
y = dataset["anomaly"]


# First normalize the individual data points
dim = X.shape[1]
n = X.shape[0]


# the following bounds are chosen independent to the data
x_bound = 1
y_bound = 1

X = x_bound*preprocessing.normalize(X, norm='l2')

def CE(score,y):
    # numerically efficient vectorized implementation of CE loss
    log_phat = np.zeros_like(score)
    log_one_minus_phat = np.zeros_like(score)
    mask = score > 0 
    log_phat[mask] = - np.log( 1 + np.exp(-score[mask]))
    log_phat[~mask] = score[~mask] - np.log( 1 + np.exp(score[~mask]))
    log_one_minus_phat[mask] = -score[mask] - np.log( 1 + np.exp(-score[mask]))
    log_one_minus_phat[~mask] = - np.log( 1 + np.exp(score[~mask]))
    
    return -y*log_phat-(1-y)*log_one_minus_phat




def loss(theta):
    return np.sum(CE(X@theta,y))/n

def err(theta):
    return np.sum((X@theta > 0) != y) / n

def err_yhat(yhat):
    return np.sum((yhat != y)) / n


clf = LogisticRegression(penalty = 'l2', random_state=0, fit_intercept=False).fit(X, y)
yhat = clf.predict(X)

err_nonprivate = err_yhat(yhat)
err_trivial = min(np.mean(y), 1-np.mean(y))
theta_values = clf.coef_[0]  # Get the coefficients

# Print results
print('Nonprivate error rate is', err_nonprivate)
print('Trivial error rate is', err_trivial)
print('Theta values:', theta_values)






from autodp.autodp_core import Mechanism
from autodp.mechanism_zoo import GaussianMechanism
from autodp.calibrator_zoo import eps_delta_calibrator
from autodp.mechanism_zoo import ExactGaussianMechanism, PureDP_Mechanism




def CE(score, y):
    """Compute cross-entropy loss"""
    log_phat = np.zeros_like(score)
    log_one_minus_phat = np.zeros_like(score)
    mask = score > 0 
    log_phat[mask] = - np.log(1 + np.exp(-score[mask]))
    log_phat[~mask] = score[~mask] - np.log(1 + np.exp(score[~mask]))
    log_one_minus_phat[mask] = -score[mask] - np.log(1 + np.exp(-score[mask]))
    log_one_minus_phat[~mask] = - np.log(1 + np.exp(score[~mask]))
    return -y*log_phat-(1-y)*log_one_minus_phat

def loss(theta, X, y, lambda_reg):
    """Compute total loss with L2 regularization"""
    # L2 regularization directly to the loss function.
    return np.sum(CE(X@theta, y))/len(y) + (lambda_reg/2) * np.sum(theta**2)

def err(theta, X, y):
    """Compute classification error"""
    return np.sum((X@theta > 0) != y) / len(y)


#This function trains the model and adds calibrated noise to ensure differential privacy. It includes both the training phase and privacy mechanism application.
#Lambda needs to grow with n for the sensitivity to be meaningful. 

def train_private_logistic_regression(X, y, lambda_reg, epsilon, delta = 1e-6, niter=10):
    """Train logistic regression with output perturbation"""
    n, d = X.shape
    theta = np.zeros(d)

    X = x_bound*preprocessing.normalize(X, norm='l2')
    
    # Train standard logistic regression
    clf = LogisticRegression(random_state=0, fit_intercept=False).fit(X, y)
    yhat = clf.predict(X)

    err_nonprivate = err_yhat(yhat)
    err_trivial = min(np.mean(y), 1-np.mean(y))
    theta_values = clf.coef_[0]  # Get the coefficients

    calibrate = eps_delta_calibrator()
    eps = epsilon
    delta = delta

    mech1 = calibrate(ExactGaussianMechanism,eps,delta,[0,100],name='GM1')
        
    sigma = mech1.params['sigma']


    noise = np.random.normal(0, sigma/lambda_reg, size=d)
    private_theta = theta_values + noise    


    return private_theta, err(private_theta, X, y)

def calculate_lambda(n, epsilon, d, L, theta_star_norm):
    delta = 1e-6  # Fixed privacy parameter
    numerator = (d**(1/3)) * (delta**(1/3)) * (L**(2/3)) * \
                (theta_star_norm**(4/3)) * (np.log(1/delta)**(1/3))
    denominator = (n**(2/3)) * (epsilon**(2/3))
    return n*(numerator / denominator)


epsilon = 1
n = X.shape[0]  
d = X.shape[1]  
L = 1.0
theta_star_norm = 5.0

lambda_value = calculate_lambda(n, epsilon, d, L, theta_star_norm)



train_private_logistic_regression(X, y, lambda_reg = lambda_value, epsilon =1)



   

epsilons = [0.5, 1.0, 1.5, 2.0]
base_lambda = lambda_value
multipliers = [1/4, 1/2, 1, 2, 4,8,16,32,64]
lambdas = [base_lambda * m for m in multipliers]
delta = 1e-6
n_runs = 100

# Create the plot
plt.figure(figsize=(12, 8))
colors = ['blue', 'orange', 'green', 'red']
epsilon_colors = dict(zip(epsilons, colors))
optimal_lambdas = []

# Plot a line for each epsilon value
for epsilon in epsilons:
    # Initialize array to store errors for each lambda
    all_errors = np.zeros((n_runs, len(lambdas)))
    
    for run in range(n_runs):
        for i, lambda_reg in enumerate(lambdas):
            _, error = train_private_logistic_regression(X, y, lambda_reg, epsilon, delta,niter = 1)
            all_errors[run, i] = error
    
    # Calculate average errors across runs
    avg_errors = np.mean(all_errors, axis=0)
    
    # Plot average line
    plt.plot(lambdas, avg_errors, color=epsilon_colors[epsilon], label=f'ε = {epsilon}')
    
    # Find and plot optimal point
    min_error_idx = np.argmin(avg_errors)
    optimal_lambda = lambdas[min_error_idx]
    optimal_lambdas.append(optimal_lambda)
    plt.scatter(optimal_lambda, avg_errors[min_error_idx], marker='*', s=100, color=epsilon_colors[epsilon])

plt.xscale('log')
plt.xlabel('Lambda (log scale)')
plt.ylabel('Average Error Rate')
plt.title('Average Error Rate vs Lambda for Different Epsilon Values')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

from autodp.autodp_core import Mechanism
from autodp.mechanism_zoo import GaussianMechanism

class LogisticRegressionOPLambdaI(Mechanism):
    def __init__(self, n, lambda_reg, epsilon, name='LogisticRegressionOP'):
        Mechanism.__init__(self)
        self.name = name
        self.params = {'n': n, 'lambda': lambda_reg, 'epsilon': epsilon}
        beta = (n * lambda_reg * epsilon)/2
        gm = GaussianMechanism(1/beta, name='Perturb_output')
        self.set_all_representation(gm)

def train_private_logistic_regression2(X, y, lambda_reg, epsilon, niter=1000):
    """Train logistic regression with output perturbation"""
    n, d = X.shape
    theta = np.zeros(d)
    
    # Add regularization directly to X^T X
    XTX = X.T @ X + lambda_reg * np.eye(d)
    
    # Train standard logistic regression
    for _ in range(niter):
        z = X @ theta
        pred = 1/(1 + np.exp(-z))
        # Use regularized XTX in gradient computation
        grad = (XTX @ theta - X.T @ y)/n
        theta = theta - 0.1 * grad
    
    # Add noise calibrated to sensitivity
    mechanism = LogisticRegressionOPLambdaI(n, lambda_reg, epsilon)
    noise_scale = 2/(n * lambda_reg * epsilon)
    noise = np.random.normal(0, noise_scale, size=d)
    private_theta = theta + noise
    
    return private_theta, err(private_theta)

epsilons = [0.5, 1.0, 1.5, 2.0]
delta = 1e-6
n_runs = 50
# Initialize storage for errors
all_errors = {eps: [] for eps in epsilons}

# Run multiple times for each epsilon with its corresponding optimal lambda
for epsilon, optimal_lambda in zip(epsilons, optimal_lambdas):
    for _ in range(n_runs):
        private_theta, error = train_private_logistic_regression(X, y, optimal_lambdas[3], epsilon, delta, niter = 2)
        all_errors[epsilon].append(error)

# Calculate average errors
avg_errors = [np.mean(all_errors[eps]) for eps in epsilons]
std_errors = [np.std(all_errors[eps]) for eps in epsilons]

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(epsilons, avg_errors, 'b-o')
plt.fill_between(epsilons, 
                 [avg - std for avg, std in zip(avg_errors, std_errors)],
                 [avg + std for avg, std in zip(avg_errors, std_errors)],
                 alpha=0.2)
plt.xlabel('Epsilon (ε)')
plt.ylabel('Average Error Rate')
plt.title(f'Average Error Rate vs Epsilon at Optimal Lambda Values ({n_runs} runs)')
plt.grid(True)

# Print results
print("\nResults averaged over 10 runs:")
for eps, avg, std in zip(epsilons, avg_errors, std_errors):
    print(f"ε = {eps:.1f}: Error = {avg:.4f} ± {std:.4f}")

plt.show()
