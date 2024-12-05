import numpy as np
import pandas as pd

from os import path
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

from typing import Callable, Tuple, Optional
from numpy.linalg import norm

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from autodp import mechanism_zoo, calibrator_zoo
from autodp.transformer_zoo import ComposeGaussian

class FTRLM():
    def __init__(self, fp, epsilons, delta):
        #switch below with parent 
        df=pd.read_csv(fp, index_col='segment')
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        self.epsilons = epsilons
        self.delta = delta

        # Set the Seaborn style
        #sns.set_theme()
        # Get the "flare" palette
        self.color_palette = sns.color_palette("flare")
        
        df = self.clean(df)
        
        self.err_nonprivate = 0
        self.err_trivial = 0
        
        self.non_private_log_reg()

        self.lamda = 0.01
        self.batch_size = 100
        self.momentum = 0.85
        self.epochs = 20
        self.batch_size = 100
        self.L = 1.0 # clip norm

        #best_params, results = self.tune_hyperparameters()
        #self.best_params = best_params
        #self.results = results

        #self.plot_eps_performance()

    def run_all_plots(self):
        self.plot_eps_performance(self.lamda, self.batch_size, self.momentum)
        

    def clean(self, df):
        #convert categorical column to multiple binary
        df=pd.get_dummies(df,columns=["channel"])

        X = df[[x for x in df.columns if x!="anomaly"]]
        y = df["anomaly"].to_numpy()

        # First normalize the individual data points
        X = preprocessing.normalize(X, norm='l2')
        
        self.X = X
        self.y = y
        self.dim = self.X.shape[1]
        self.n = self.X.shape[0]

    def err(self, X, y, theta):
        return np.sum((X @theta > 0) != y) / X.shape[0]

    def err_yhat(self, y, yhat):
        return np.sum((yhat != y)) / len(y)

    def logistic_loss_grad(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute per-example gradients for logistic regression
        
        Args:
            X: Input features of shape (batch_size, dim)
            y: Labels of shape (batch_size,)
            theta: Model parameters of shape (dim,)
        
        Returns:
            Gradients of shape (batch_size, dim)
        """
        # Compute predictions
        z = X @ theta  # Shape: (batch_size,)
        sigmoid = 1 / (1 + np.exp(-z))  # Shape: (batch_size,)
        
        # Compute per-example gradients
        grads = X * (sigmoid - y)[:, np.newaxis]  # Shape: (batch_size, dim)
        return grads

    def non_private_log_reg(self):
        clf = LogisticRegression(random_state=0,fit_intercept=False).fit(self.X, self.y)
        yhat = clf.predict(self.X)

        self.err_nonprivate = self.err_yhat(self.y, yhat)
        self.err_trivial = min(np.mean(self.y), 1-np.mean(self.y))
    
    def run_tune_hyperparameters(self, X, y, dim, num_examples, l2_clip, 
                        target_epsilon, target_delta, epochs):
        # Grid for hyperparameters
        momentums = [0.9, 0.97, 0.85, 0.7]
        batch_sizes = [100, 200, 150]
        lambdas = [0.001, 0.01, 0.1, 1.0]
        
        best_error = float('inf')
        best_params = None
        results = []
        
        for m in momentums:
            for lamb in lambdas:
                adjusted_lr = 1 / (lamb * num_examples)
                for bs in batch_sizes:
                    optimizer = DPFTRLM(
                        dim=dim,
                        num_examples=num_examples,
                        batch_size=bs,
                        learning_rate=adjusted_lr,
                        momentum=m,
                        l2_norm_clip=l2_clip,
                        lambda_reg=lamb,
                        target_epsilon=target_epsilon,
                        target_delta=target_delta,
                        epochs=epochs,
                        loss_grad_fn=self.logistic_loss_grad
                    )
                    
                    # Training loop
                    theta = np.zeros(optimizer.dim)
                    final_error = float('inf')
                    
                    for epoch in range(optimizer.epochs):
                        theta, eps, error = optimizer.train_epoch(X, y, theta)
                        final_error = error
                    
                    results.append({
                        'momentum':m,
                        'batch_size': bs,
                        'lambda': lamb,
                        'final_error': final_error
                    })
                    
                    if final_error < best_error:
                        best_error = final_error
                        best_params = {'batch_size': bs, 'momentum': m, 'lambda': lamb}
                        
                    #print(f"momentum: {m:.2f}, batch_size: {bs}, lambda: {lamb:.6f}, error: {final_error:.4f}")
        
        #print("\nBest parameters:")
        ##print(f"Momentum: {best_params['momentum']:.2f}")
        #print(f"batch_size: {best_params['batch_size']}")
        #print(f"Lambda: {best_params['lambda']:.6f}")
        #print(f"Best error: {best_error:.4f}")
        
        return best_params, results
    
    def tune_hyperparameters(self):
        # Usage example
        best_params, results = self.run_tune_hyperparameters(
            X=self.X,
            y=self.y,
            dim=self.dim,
            num_examples=self.n,
            #batch_size=100,
            #momentum=0.9,
            l2_clip=1.0,
            target_epsilon=100,
            target_delta=1e-5,
            epochs=50
        )
        return best_params, results
    
    def plot_eps_performance(self, lamb, batch_size, momentum, epochs, L):
        best_errs = []
        epss = self.epsilons
        for test_eps in epss:
            # Initialize optimizer
            optimizer = DPFTRLM(
                dim=self.dim,
                num_examples=self.n,
                batch_size=batch_size,
                learning_rate=1 / (lamb * self.n), # 1 / lambda?
                momentum=momentum,
                l2_norm_clip=L,
                lambda_reg=lamb, # Add regularization strength
                target_epsilon=test_eps,
                target_delta=self.delta,
                epochs=epochs,
                loss_grad_fn=self.logistic_loss_grad
            )

            # Training loop
            theta = np.zeros(optimizer.dim)
            for epoch in range(optimizer.epochs):
                theta, eps, error = optimizer.train_epoch(self.X, self.y, theta)
                #if epoch % 10 == 0:
                #    print(f"Epoch {epoch}, ε = {eps}")

            #if abs(optimizer.get_eps()) > .01:
                #print(optimizer.get_eps())
            
            best_errs.append(error)

        plt.figure(figsize=(8, 5))
        plt.plot(epss,np.array(best_errs),self.color_palette[0])
        plt.plot(epss,self.err_nonprivate*np.ones_like(epss),self.color_palette[2])
        plt.plot(epss,self.err_trivial*np.ones_like(epss),self.color_palette[4])

        plt.legend(['DP_FTRLM','Nonprivate','trivial'])
        plt.xlabel('Epsilon')
        plt.ylabel('Error')
        plt.savefig(f"plots/FTRLM.png")

        return pd.DataFrame({'Epsilon':epss, 'Error':best_errs, 'Method':['FTRLM' for _ in range(len(epss))]})

class DPFTRLPrivacyEngine:
    """Privacy engine for DP-FTRL with tree restart"""
    def __init__(self, total_epsilon: float, epochs: int, delta: float, height: int):
        self.total_epsilon = total_epsilon
        self.epochs = epochs
        self.delta = delta
        self.height = height
        
        # Allocate epsilon budget equally across epochs due to tree restart
        self.epsilon_per_epoch = total_epsilon / epochs
        
        # Initialize calibrator
        self.calibrator = calibrator_zoo.ana_gaussian_calibrator()
        
    def get_sigma(self, l2_clip: float) -> float:
        """Get optimal sigma for noise calibration"""
        mechanism = self.calibrator.calibrate(
            mech_class=mechanism_zoo.ExactGaussianMechanism,
            eps=self.epsilon_per_epoch,
            delta=self.delta/self.epochs
        )
        return mechanism.params['sigma'] * l2_clip * np.sqrt(self.height)

class DPTreeAggregator:
    """Binary tree structure for private gradient aggregation"""
    def __init__(self, num_steps: int, dim: int, sigma: float):
        self.height = int(np.ceil(np.log2(num_steps))) + 1
        self.dim = dim
        self.tree = np.zeros((2**self.height - 1, dim))
        self.sigma = sigma
        self.noise = np.zeros((2**self.height - 1, dim))
        
    def add_gradient(self, step: int, gradient: np.ndarray) -> None:
        idx = 2**(self.height - 1) + step
        self.tree[idx] = gradient
        
        while idx > 0:
            parent = (idx - 1) // 2
            if idx % 2 == 1:
                self.tree[parent] = self.tree[parent - 1] + self.tree[idx]
            else:
                self.tree[parent] = self.tree[idx]
            idx = parent
            
    def get_noisy_sum(self, step: int) -> np.ndarray:
        total = np.zeros(self.dim)
        idx = 2**(self.height - 1) + step
        #print(self.sigma)
        while idx > 0:
            parent = (idx - 1) // 2
            if idx % 2 == 1:
                if (self.noise[parent] == np.zeros(self.dim)).any():
                    self.noise[parent] = np.random.normal(0, self.sigma, self.dim)
                total += self.tree[idx] + self.noise[parent]
            idx = parent
        return total

class DPFTRLM:
    def __init__(
        self,
        dim: int,
        num_examples: int,
        batch_size: int,
        learning_rate: float,
        momentum: float,
        l2_norm_clip: float,
        lambda_reg: float,
        target_epsilon: float,
        target_delta: float,
        epochs: int,
        loss_grad_fn: Callable
    ):
        self.dim = dim
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.lr = learning_rate
        self.momentum = momentum
        self.l2_clip = l2_norm_clip
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.loss_grad_fn = loss_grad_fn
        
        # Initialize privacy engine
        self.privacy_engine = DPFTRLPrivacyEngine(
            total_epsilon=target_epsilon,
            epochs=epochs,
            delta=target_delta,
            height= int(np.ceil(self.num_examples / self.batch_size))
        )
        
        # Initialize momentum buffer and initial point
        self.velocity = np.zeros(dim)
        self.theta_0 = np.zeros(dim)

    def get_eps(self):
        return self.privacy_engine.total_epsilon
        
    def train_epoch(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, float]:
        num_batches = int(np.ceil(self.num_examples / self.batch_size))
        
        # Initialize tree for this epoch with calibrated noise
        tree = DPTreeAggregator(
            num_steps=num_batches,
            dim=self.dim,
            sigma=self.privacy_engine.get_sigma(self.l2_clip)
        )
        
        perm = np.random.permutation(self.num_examples)
        X, y = X[perm], y[perm]
        
        for batch in range(num_batches):
            start_idx = batch * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_examples)
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            grads = self.loss_grad_fn(X_batch, y_batch, theta)
            grad_norms = np.linalg.norm(grads, axis=1)
            grad_norms = np.maximum(grad_norms, 1e-12)  # Prevent division by zero
            scaling = np.minimum(1, self.l2_clip / grad_norms)
            grads = grads * scaling[:, np.newaxis]
            
            tree.add_gradient(batch, np.mean(grads, axis=0))
            
            noisy_grads = tree.get_noisy_sum(batch)
            
            # Update with momentum and regularization
            self.velocity = self.momentum * self.velocity + noisy_grads
            reg_grad = self.lambda_reg * (theta - self.theta_0)
            theta = theta - self.lr * (self.velocity + reg_grad)
            
        # With tree restart, privacy spent per epoch is fixed
        privacy_spent = self.privacy_engine.epsilon_per_epoch
        self.privacy_engine.total_epsilon -= privacy_spent
        error = self.err(X, y, theta)
        
        return theta, privacy_spent, error
    




#ftlrm_obj = FTRLM(fp=data['fp'], epsilons=data['epsilons'], delta=data['delta'])
#objpert_df = ftlrm_obj.run_all_plots()
