"""Runs tests on the non-private Frank-Wolfe Lasso implementation varying parameters and saves the plots."""
# This code is based on implementation_LASSO.py from public repository: 
# "Frank-Wolfe Algorithm in Python" by Le Anh DUNG and Paul MELKI (Toulouse School of Economics)
# https://github.com/paulmelki/Frank-Wolfe-Algorithm-Python

#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from src.model_build.frank_wolfe_deprecated import frankWolfeLASSO
# Todo is edit FW_NonPrivate to have these same outputs!
#from src.model_build.frankWolfeLASSO import FW_NonPrivate


#%% First round of testing of our implementation:
# We fix the number of obseverations (n) and the number of parameters (p)
# and study the performance of the algorithm when only lambda (l) is changing.

# Array holding the tolerance levels we will be working with
tolerances = np.array([0.0001, 0.001, 0.01, 0.1])

# Pandas dataframe that will hold the data used to create the plot
dataToPlot = pd.DataFrame({
    "tolerances" : np.array(tolerances),
    "k1" : np.zeros(len(tolerances)),
    "k2" : np.zeros(len(tolerances)),
    "k3" : np.zeros(len(tolerances))
})

# constant parameters
n = 1000 # number of observations
p = 700 # number of parameters
max_iter = 15_000 # stop if not converged by
# Array that will hold the returned numbers of iterations for each level of 
# tolerance.
returnedK = np.zeros(4)
# Array that will hold the returned data (f(x))
data = [None] * 4
# Array that will hold the norm of the difference: ||f(x_k) - (f_(k-1))
diffx = [None] * 4
   
# CASE 1
l1 = 50    # penalty parameter
A, y = datasets.make_regression(n, p)
y = y.reshape((n, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k1'] = \
      frankWolfeLASSO(A, y, l=l1, tol=tolerances[i])
# replace with new model eventually:
# FW_NonPrivate(A, y, l=l1, K=max_iter, tol=tolerances[i], trace=True, normalize=False, clip_sd=None)


# CASE 2:
returnedK = np.zeros(4)
data = [None] * 4
diffx = [None] * 4


l2 = 500
A, y = datasets.make_regression(n, p)
y = y.reshape((n, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k2'] = \
        frankWolfeLASSO(A, y, l=l2, tol=tolerances[i])


# CASE 3: l = 5000, n = 1000, p = 700
returnedK = np.zeros(4)
data = [None] * 4
diffx = [None] * 4
l3 = 5000
A, y = datasets.make_regression(n, p)
y = y.reshape((n, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k3'] = \
        frankWolfeLASSO(A, y, l=l3, tol=tolerances[i])

# Plot the results
plt.plot(dataToPlot.tolerances, dataToPlot.k1, marker="o", color="darkgreen",\
              label = f"l = {l1}, n = {n}, p = {p}")
plt.plot(dataToPlot.tolerances, dataToPlot.k2, marker="o", color="deepskyblue", \
              label = f"l = {l2}, n = {n}, p = {p}")
plt.plot(dataToPlot.tolerances, dataToPlot.k3, marker="o", color="firebrick", \
              label = f"l = {l3}, n = {n}, p = {p}")
plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle("Solving LASSO problem using Frank-Wolfe Algorithm", fontsize=16)
plt.xlabel("Tolerance level")
plt.ylabel("Number of iterations (k)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("Iterations_l.png")
plt.show()



#%% Second round of testing of our implementation:
# We fix the number of obseverations (n) and lambda (l)
# and study the performance of the algorithm when only p is changing.

# can change tolerances here if you want
# tolerances = np.array([0.1, 0.01, 0.001, 0.0001])

# Pandas dataframe that will hold the data used to create the plot
dataToPlot = pd.DataFrame({
    "tolerances" : np.array(tolerances),
    "k1" : np.zeros(len(tolerances)),
    "k2" : np.zeros(len(tolerances)),
    "k3" : np.zeros(len(tolerances))
})
# constant parameters
n = 1000 
l = 50
# Array that will hold the returned numbers of iterations for each level of 
# tolerance.
returnedK = np.zeros(4)
# Array that will hold the returned data (f(x))
data = [None] * 4
# Array that will hold the norm of the difference: ||f(x_k) - (f_(k-1))
diffx = [None] * 4

# CASE 1
p1 = 700   # number of parameters
A, y = datasets.make_regression(n, p1)
y = y.reshape((n, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k1'] = \
      frankWolfeLASSO(A, y, l=l, tol=tolerances[i])


# CASE 2
p2 = 1400
A, y = datasets.make_regression(n, p2)
y = y.reshape((n, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k2'] = \
        frankWolfeLASSO(A, y, l=l, tol=tolerances[i])


# CASE 3
p3 = 7000
A, y = datasets.make_regression(n, p3)
y = y.reshape((n, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k3'] = \
        frankWolfeLASSO(A, y, l=l, tol=tolerances[i])

# Plot the results
plt.plot(dataToPlot.tolerances, dataToPlot.k1, marker = "o", color = "darkslategray",\
              label = f"n = {n}, p = {p1}, l = {l}")
plt.plot(dataToPlot.tolerances, dataToPlot.k2, marker = "o", color = "chartreuse", \
              label = f"n = {n}, p = {p2}, l = {l}")
plt.plot(dataToPlot.tolerances, dataToPlot.k3, marker = "o", color = "slateblue", \
              label = f"n = {n}, p = {p3}, l = {l}")
plt.legend()
plt.suptitle("Solving LASSO problem using Frank-Wolfe Algorithm", fontsize=16)
plt.xlabel("Tolerance level")
plt.ylabel("Number of iterations (k)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("Iterations_p.png")
plt.show()



#%% Third round of testing of our implementation:
# We fix the number of parameters (p) and lambda (l)
# and study the performance of the algorithm when only n is changing.

# can change tolerances here if you want
# tolerances = np.array([0.1, 0.01, 0.001, 0.0001])

# Pandas dataframe that will hold the data used to create the plot
dataToPlot = pd.DataFrame({
    "tolerances" : np.array(tolerances),
    "k1" : np.zeros(len(tolerances)),
    "k2" : np.zeros(len(tolerances)),
    "k3" : np.zeros(len(tolerances))
})

# Array that will hold the returned numbers of iterations for each level of 
# tolerance.
returnedK = np.zeros(4)
# Array that will hold the returned data (f(x))
data = [None] * 4
# Array that will hold the norm of the difference: ||f(x_k) - (f_(k-1))
diffx = [None] * 4

# constant parameters
p = 1400   # number of parameters
l = 50    # penalty parameter

# CASE 1
n1 = 1000  # number of observations
A, y = datasets.make_regression(n1, p)
y = y.reshape((n1, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k1'] = \
      frankWolfeLASSO(A, y, l=l, tol=tolerances[i])


# CASE 2
n2 = 5000
A, y = datasets.make_regression(n2, p)
y = y.reshape((n2, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k2'] = \
        frankWolfeLASSO(A, y, l=l, tol=tolerances[i])


# CASE 3
n3 = 10000
A, y = datasets.make_regression(n3, p)
y = y.reshape((n3, 1))

for i in range(4):
    data[i], diffx[i], dataToPlot.loc[i, 'k3'] = \
        frankWolfeLASSO(A, y, l=l, tol=tolerances[i])

# Plot the results
plt.plot(dataToPlot.tolerances, dataToPlot.k1, marker = "o", color = "cadetblue",\
              label = f"n = {n1}, p = {p}, l = {l}")
plt.plot(dataToPlot.tolerances, dataToPlot.k2, marker = "o", color = "darkorange", \
              label = f"n = {n2}, p = {p}, l = {l}")
plt.plot(dataToPlot.tolerances, dataToPlot.k3, marker = "o", color = "midnightblue", \
              label = f"n = {n3}, p = {p}, l = {l}")
plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle("Solving LASSO problem using Frank-Wolfe Algorithm", fontsize=16)
plt.xlabel("Tolerance level")
plt.ylabel("Number of iterations (k)")
plt.savefig("Iterations_n.png")
plt.show()
