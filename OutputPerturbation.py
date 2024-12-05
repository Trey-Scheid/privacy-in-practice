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


class OutputPerturbation():
    def __init__(self, fp, epsilons, delta):
        self.eps = epsilons
        self.delta = delta
        self.fp = fp
        (
            self.X,
            self.y,
            self.dim,
            self.n,
            self.x_bound,
            self.y_bound,
        ) = self.clean_data()
        self.df = None
        self.base_lambda = self.calculate_base_lambda()
        


    def clean_data(self):
        df = pd.read_csv(self.fp)

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

        return (X,y,dim,n,x_bound,y_bound)
    
    def CE(self,score,y):
            # numerically efficient vectorized implementation of CE loss
            log_phat = np.zeros_like(score)
            log_one_minus_phat = np.zeros_like(score)
            mask = score > 0 
            log_phat[mask] = - np.log( 1 + np.exp(-score[mask]))
            log_phat[~mask] = score[~mask] - np.log( 1 + np.exp(score[~mask]))
            log_one_minus_phat[mask] = -score[mask] - np.log( 1 + np.exp(-score[mask]))
            log_one_minus_phat[~mask] = - np.log( 1 + np.exp(score[~mask]))
        
            return -y*log_phat-(1-y)*log_one_minus_phat





    def loss(self, theta, X, y, lambda_reg):
            """Compute total loss with L2 regularization"""
            # L2 regularization directly to the loss function.
            return np.sum(self.CE(X@theta, y))/len(y) + (lambda_reg/2) * np.sum(theta**2)

    def err(self,theta,X,y):
            return np.sum((X@theta > 0) != y) / len(y)

    def err_yhat(self, yhat):
            return np.sum((yhat != self.y)) / self.n


        







    def train_private_logistic_regression(self, X, y, lambda_reg, epsilon, delta, niter):
        """Train logistic regression with output perturbation"""
        n, d = X.shape
        
        # Train standard logistic regression
        clf = LogisticRegression(random_state=0, fit_intercept=False).fit(X, y)
        yhat = clf.predict(X)
        theta_values = clf.coef_[0]

        calibrate = eps_delta_calibrator()
        mech1 = calibrate(ExactGaussianMechanism, epsilon, delta, [0,100], name='GM1')
        sigma = mech1.params['sigma']

        noise = np.random.normal(0, sigma/lambda_reg, size=d)
        private_theta = theta_values + noise    

        return private_theta, self.err(private_theta, X, y)

    def calculate_lambda(self, n, epsilon, d, L, theta_star_norm):
            delta = 1e-6  # Fixed privacy parameter
            numerator = (d**(1/3)) * (delta**(1/3)) * (L**(2/3)) * \
                        (theta_star_norm**(4/3)) * (np.log(1/delta)**(1/3))
            denominator = (n**(2/3)) * (epsilon**(2/3))
            return n*(numerator / denominator)


        
    def calculate_base_lambda(self):
        lambda_value = self.calculate_lambda(self.n, 1, self.dim, 1, 5)
        return lambda_value





    def run_experiment(self):
        multipliers = [1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]
        lambdas = [self.base_lambda * m for m in multipliers]
        n_runs = 100
        optimal_lambdas = []

        plt.figure(figsize=(12, 8))
        colors = ['blue', 'orange', 'green', 'red']
        epsilon_colors = dict(zip(self.eps, colors))

        for epsilon in self.eps:
            all_errors = np.zeros((n_runs, len(lambdas)))
            for run in range(n_runs):
                for i, lambda_reg in enumerate(lambdas):
                    _, error = self.train_private_logistic_regression(
                        self.X, self.y, lambda_reg, epsilon, self.delta, niter=1
                    )
                    all_errors[run, i] = error
            
            avg_errors = np.mean(all_errors, axis=0)
            plt.plot(lambdas, avg_errors, color=epsilon_colors[epsilon], 
                    label=f'ε = {epsilon}')
            
            min_error_idx = np.argmin(avg_errors)
            optimal_lambda = lambdas[min_error_idx]
            optimal_lambdas.append(optimal_lambda)
            plt.scatter(optimal_lambda, avg_errors[min_error_idx], 
                    marker='*', s=100, color=epsilon_colors[epsilon])

        plt.xscale('log')
        plt.xlabel('Lambda (log scale)')
        plt.ylabel('Average Error Rate')
        plt.title('Average Error Rate vs Lambda for Different Epsilon Values')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/OutputPerturbationErrorvsLambda.png')
        
        return optimal_lambdas

    def plot_epsilon_error(self, optimal_lambdas, n_runs=50):
        all_errors = {eps: [] for eps in self.eps}
        
        for epsilon, optimal_lambda in zip(self.eps, optimal_lambdas):
            for _ in range(n_runs):
                _, error = self.train_private_logistic_regression(
                    self.X, self.y, optimal_lambda, epsilon, self.delta, niter=2
                )
                all_errors[epsilon].append(error)
        
        avg_errors = [np.mean(all_errors[eps]) for eps in self.eps]
        std_errors = [np.std(all_errors[eps]) for eps in self.eps]
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.eps, avg_errors, 'b-o')
        plt.fill_between(self.eps, 
                        [avg - std for avg, std in zip(avg_errors, std_errors)],
                        [avg + std for avg, std in zip(avg_errors, std_errors)],
                        alpha=0.2)
        plt.xlabel('Epsilon (ε)')
        plt.ylabel('Average Error Rate')
        plt.title(f'Average Error Rate vs Epsilon at Optimal Lambda Values ({n_runs} runs)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/OutputPerturbationEpsilonvsError.png')
        
        return avg_errors, std_errors
    

epsilons = [0.5, 1.0, 1.5, 2.0]
delta = 1e-6
op = OutputPerturbation("data/dataset.csv", epsilons, delta)
optimal_lambdas = op.run_experiment()
avg_errors, std_errors = op.plot_epsilon_error(optimal_lambdas)