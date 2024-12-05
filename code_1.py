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






