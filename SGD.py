import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from autodp.autodp_core import Mechanism
from autodp.autodp_core import Mechanism
from autodp.mechanism_zoo import ExactGaussianMechanism
from autodp.transformer_zoo import AmplificationBySampling, Composition
from autodp.calibrator_zoo import eps_delta_calibrator
import seaborn as sns
class SGD():
    def clean(self,df):
        df=pd.get_dummies(df,columns=["channel"])
        dataset=df
        self.X = dataset[[x for x in dataset.columns if x!="anomaly"]]
        self.X = 1*preprocessing.normalize(self.X, norm='l2')
        self.y = dataset["anomaly"]
        self.dim = self.X.shape[1]
        self.n = self.X.shape[0]
    def regSGD(self):
        clf = SGDClassifier(random_state=0,fit_intercept=False).fit(self.X, self.y)
        yhat = clf.predict(self.X)
        self.err_nonprivate = self.err_yhat(yhat)
        self.err_trivial = min(np.mean(self.y), 1-np.mean(self.y) )

    def __init__(self):
        #switch below with parent 
        df=pd.read_csv('data/dataset.csv')
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.clean(df)
        #switch below with parent
        self.delta=1e-6
        self.GS=1
        self.regSGD()
        
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
    def loss(self,theta):
        return np.sum(self.CE(self.X@theta,self.y))/self.n

    def err(self,theta):
        return np.sum((self.X@theta > 0) != self.y) / self.n

    def err_yhat(self,yhat):
        return np.sum((yhat != self.y)) / self.n
    class NoisySGD_Mechanism_With_Amplification(Mechanism):
        def __init__(self, prob, sigma, niter, PoissonSampling=True, name='NoisySGD_with_amplification'):
            """
            A Noisy SGD mechanism with subsampling amplification.
            Args:
                prob: Sampling probability (batch_size / dataset_size).
                sigma: Noise scale for the Gaussian mechanism.
                niter: Number of SGD iterations.
                PoissonSampling: Whether Poisson sampling is used (default: True).
            """
            Mechanism.__init__(self)
            self.name = name
            self.params = {'prob': prob, 'sigma': sigma, 'niter': niter,
                        'PoissonSampling': PoissonSampling}

            subsample = AmplificationBySampling(PoissonSampling=PoissonSampling)

            gm = ExactGaussianMechanism(sigma=sigma)

            subsampled_mech = subsample(gm, prob, improved_bound_flag=True)

            compose = Composition()
            mech = compose([subsampled_mech], [niter])

            rdp_total = mech.RenyiDP

            self.propagate_updates(rdp_total, type_of_update='RDP')   
    def stochastic_gradient(self,theta, X_batch, y_batch):
        phat = np.exp(X_batch @ theta) / (1 + np.exp(X_batch @ theta))
        grad = X_batch.T @ (phat - y_batch)
        return grad
    def GS_bound(self,theta):
        """
        Calculate global sensitivity for a mini-batch.
        """
        GS = 100
        bound = np.linalg.norm(theta)
        GS = self.GS / (1 + np.exp(-bound))
        return GS
        
    def run_NoisySGD_step(self,theta, sigma, lr, X_batch, y_batch):
        """
        Perform one step of Noisy SGD.
        """
        GS = self.GS_bound(theta)
        noisy_gradient = self.stochastic_gradient(theta, X_batch, y_batch) + GS*sigma*np.random.normal(size=theta.shape)
        return theta - lr * noisy_gradient
    def run_NoisySGD(self,X, y, sigma, lr, niter, batch_size, log_gap=10):
        """
        Run stochastic gradient descent with noise for privacy.
        """
        theta_SGD = np.zeros(shape=(self.dim,))
        err_SGD = []
        eps_SGD = []
        prev=theta_SGD.copy()
        for i in range(niter):
            # Randomly sample a mini-batch
            idx = np.random.choice(len(y), batch_size, replace=False)
            X_batch = X[idx, :]
            y_batch = y[idx]
            
            # Perform a noisy SGD step
            theta_SGD = self.run_NoisySGD_step(theta_SGD, sigma, lr, X_batch, y_batch)
            prev=((prev*i)/(i+1))+(theta_SGD/(i+1))
            
            # Log error and privacy loss at intervals
            if i!=0:
                if not i % log_gap:
                    mech = self.NoisySGD_Mechanism_With_Amplification(prob=batch_size / len(y), sigma=sigma, niter=i + 1)
                    eps_SGD.append(mech.approxDP(self.delta))
                    #err_SGD.append(err(theta_SGD))
                    err_SGD.append(self.err(prev))
        
        return err_SGD, eps_SGD

    # function to run NoisyGD 
    def run_nonprivate_SGD(self,X, y, lr, niter, batch_size, log_gap):
        theta_SGD = np.zeros(shape=(self.dim,))
        prev=theta_SGD.copy()
        err_SGD = []
        for i in range(niter):
            # Randomly sample a mini-batch
            idx = np.random.choice(len(y), batch_size, replace=False)
            X_batch = X[idx, :]
            y_batch = y[idx]
            
            # Perform a noisy SGD step
            theta_SGD = self.run_NoisySGD_step(theta_SGD, 0, lr, X_batch, y_batch)
            prev=((prev*i)/(i+1))+(theta_SGD/(i+1))
            
            # Log error and privacy loss at intervals
            if i!=0:
                if not i % log_gap:
                    err_SGD.append(self.err(prev))
        
        return err_SGD


    def find_appropriate_niter(self,sigma, eps,delta,num):
        # Use autodp calibrator for selecting 'niter'
        NoisyGD_fix_sigma = lambda x:  self.NoisySGD_Mechanism_With_Amplification(num/self.n, sigma, x)
        calibrate = eps_delta_calibrator()
        mech = calibrate(NoisyGD_fix_sigma, eps, delta, [1,500000])
        niter = int(np.floor(mech.params['niter']))
        return niter
    def theoretical_lr_choice(self,beta_L,f0_minus_fniter_bound,dim,sigma,niter):
        return np.minimum(1/beta_L,np.sqrt(2*f0_minus_fniter_bound / (dim * sigma**2 *beta_L*niter)))
    def diffNoisePlot(self,epsilon):
        beta = 1/4*self.n

        f0_minus_fniter_bound  =  self.n*(-np.log(0.5))
        #large nosie
        sigma = 30.0
        niter = self.find_appropriate_niter(sigma, epsilon,self.delta,64)
        lr = self.theoretical_lr_choice(beta,f0_minus_fniter_bound,self.dim,sigma*self.GS,niter)

        err_SGD1, eps_SGD1 = self.run_NoisySGD(self.X,self.y,sigma,lr,niter,log_gap=3000,batch_size=64)
        self.err_SGD1=err_SGD1
        self.eps_SGD1=eps_SGD1
        # Small noise
        sigma = 3
        niter = self.find_appropriate_niter(sigma, epsilon,self.delta,64)
        lr = self.theoretical_lr_choice(beta,f0_minus_fniter_bound, self.dim,sigma*self.GS,niter)
        err_SGD2, eps_SGD2 = self.run_NoisySGD(self.X,self.y,sigma,lr,niter,log_gap=100,batch_size=64)
        #no noise baseline
        err_SGD0=self.run_nonprivate_SGD(self.X, self.y, lr, niter, 64, log_gap=100)
        sns.set(style="whitegrid", context="talk")

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(eps_SGD1, err_SGD1, 'g.-')
        plt.plot(eps_SGD2, err_SGD2, 'c.-')
        plt.plot(eps_SGD1, self.err_nonprivate * np.ones_like(eps_SGD1), 'k--')
        plt.plot(eps_SGD1, self.err_trivial * np.ones_like(eps_SGD1), 'r--')
        plt.plot(eps_SGD2, err_SGD0, 'b--')

        # Add labels, legend, and optional limits
        plt.legend([
            'NoisySGD-large-noise-more-iter', 
            'NoisySGD-small-noise-fewer-iter', 
            'Nonprivate-sklearn', 
            'Trivial', 
            'Non-private-SGD'
        ],fontsize='x-small', 
            loc='center left',               # Place it to the left of the anchor point
            bbox_to_anchor=(1, 0.5),         # Anchor the legend to the right of the plot
            ncol=1   )
        plt.xlabel('Epsilon')
        plt.ylabel('Error')
        plt.title('Error vs. Epsilon for Different NoisySGD Configurations')

        # Show plot
        plt.tight_layout()
        plt.savefig('plots/SGDNoise.png')
        plt.show()
        
    def diffLearningRatesPlot(self,epsilon):
        sigma = 30.0
        delta = 1e-6
        beta = 1/4*self.n
        GS = self.GS
        f0_minus_fniter_bound  =  self.n*(-np.log(0.5))
        niter = self.find_appropriate_niter(sigma, epsilon,delta,64)
        theoretical_lr=self.theoretical_lr_choice(beta,f0_minus_fniter_bound,self.dim,sigma*GS,niter)
        lr = 10*theoretical_lr
        err_SGD3, eps_SGD3 = self.run_NoisySGD(self.X,self.y,sigma,lr,niter,log_gap=1000,batch_size=64)
        lr = 0.1*theoretical_lr
        err_SGD4, eps_SGD4 = self.run_NoisySGD(self.X,self.y,sigma,lr,niter,log_gap=1000,batch_size=64)
        lr = 100*theoretical_lr
        err_SGD5, eps_SGD5 = self.run_NoisySGD(self.X,self.y,sigma,lr,niter,log_gap=1000,batch_size=64)
        sns.set(style="whitegrid", context="talk")
        sns.color_palette("flare")

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(self.eps_SGD1, self.err_SGD1, 'g.-')
        plt.plot(eps_SGD3, err_SGD3, 'c--')
        plt.plot(eps_SGD4, err_SGD4, 'm:')
        plt.plot(eps_SGD5, err_SGD5, 'b:')
        plt.plot(self.eps_SGD1, self.err_nonprivate * np.ones_like(self.eps_SGD1), 'k--')
        plt.plot(self.eps_SGD1, self.err_trivial * np.ones_like(self.eps_SGD1), 'r--')

        # Add legend, labels, and title
        plt.legend(
            ['NoisySGD', 'NoisySGD-lr*10', 'NoisySGD-lr/10', 'NoisySGD-lr*100', 'Nonprivate-sklearn', 'Trivial'], 
            fontsize='x-small', 
            loc='center left',               # Place it to the left of the anchor point
            bbox_to_anchor=(1, 0.5),         # Anchor the legend to the right of the plot
            ncol=1                           # Single column layout
        )
        plt.xlabel('Epsilon')
        plt.ylabel('Error')
        plt.title('Error vs. Epsilon for Noisy SGD Variants')

        # Show and save the plot
        plt.tight_layout()
        plt.savefig('plots/SGDLearningRates.png')
        plt.show()
        
    def run_NoisySGD_end(self,X, y, sigma, lr, niter, batch_size, log_gap=10):
        """
        Run stochastic gradient descent with noise for privacy and only one privacy mech at the end
        """
        theta_SGD = np.zeros(shape=(self.dim,))
        err_SGD = []
        eps_SGD = []
        prev_theta=theta_SGD.copy()
        for i in range(niter):
            # Randomly sample a mini-batch
            idx = np.random.choice(len(y), batch_size, replace=False)
            X_batch = X[idx, :]
            y_batch = y[idx]
            
            # Perform a noisy SGD step
            theta_SGD = self.run_NoisySGD_step(theta_SGD, sigma, lr, X_batch, y_batch)
            #if i>niter*.1:
            #    theta_SGD=((prev_theta*i)/(i+1))+(theta_SGD/(i+1))
            # Log error and privacy loss at intervals
            if i==niter-1:
                mech = self.NoisySGD_Mechanism_With_Amplification(prob=batch_size / len(y), sigma=sigma, niter=i + 1)
                eps_SGD.append(mech.approxDP(self.delta))
                err_SGD.append(self.err(theta_SGD))
        return err_SGD, eps_SGD

    def epsPoints(self):
        def average_runs(num_runs, X, y, ep, delta, batch_size):
            """Run NoisySGD multiple times and return the averaged results."""
            err_list = []
            eps_list = []
            niter = self.find_appropriate_niter(3, ep, delta, batch_size)
            for _ in range(num_runs):
                err, eps = self.run_NoisySGD_end(X, y, 3, 0.01, niter, log_gap=1000, batch_size=batch_size)
                err_list.append(err)
                eps_list.append(eps)

            # Compute the average across all runs
            avg_err = np.mean(err_list, axis=0)
            avg_eps = np.mean(eps_list, axis=0)
            return avg_err, avg_eps

        # Number of runs to average
        num_runs = 5

        # Run and average for different epsilon (0.5, 1.0, 1.5, 2.0)
        err_SGD_point5, eps_SGD_point5 = average_runs(num_runs, self.X, self.y, 0.5, self.delta, 64)
        err_SGD_1point0, eps_SGD_1point0 = average_runs(num_runs, self.X, self.y, 1.0, self.delta, 64)
        err_SGD_1point5, eps_SGD_1point5 = average_runs(num_runs, self.X, self.y, 1.5, self.delta, 64)
        err_SGD_2point0, eps_SGD_2point0 = average_runs(num_runs, self.X, self.y, 2.0, self.delta, 64)
        return [err_SGD_point5,err_SGD_1point0,err_SGD_1point5,err_SGD_2point0]




                            
