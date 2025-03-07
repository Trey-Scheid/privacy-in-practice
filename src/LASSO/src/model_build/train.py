from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score#, jaccard_score
import matplotlib.pyplot as plt
import numpy as np

from src.model_build import frankWolfeLASSO as FWLasso

y_name = 'power_mean'

def jaccard_similarity(set1, set2):
    intersection = len(set(set1).intersection(set2))
    union = len(set(set1).union(set2))
    return intersection / union

def trivial(feat, correct_feats):
    # prep feature data and prediction array
    X, y = feat.drop(y_name, axis=1), feat[y_name]
    ones_column = np.ones((X.shape[0], 1))
    X_train, X_test, y_train, y_test = train_test_split(np.hstack((ones_column, X)), y, test_size=0.2, random_state=1)
    
    # create trivial model
    avg = np.mean(y_train)
    coef = np.append(np.array([avg]), np.zeros(X.shape[1]))

    # compute metrics
    print(f'Train MSE sklearn: {mean_squared_error(y_train, np.repeat(avg, y_train.shape[0])):.2f} ({100*sum(coef>0)/coef.shape[0]:.1f}% sparse)')
    
    mse = mean_squared_error(y_test, np.repeat(avg, y_test.shape[0]))
    # print(f'Test MSE sklearn: {mse:.2f} ({100*sum(model.coef_>0)/model.coef_.shape[0]:.1f}% sparse)')
    
    coef_dict = dict(zip(np.append(["Intercept"], X.columns), coef))
    
    r2 = r2_score(y_test, np.repeat(avg, y_test.shape[0]), force_finite=False)
    
    similarity = jaccard_similarity(correct_feats, [k for k, v in coef_dict.items() if v > 0])
    
    return mse, coef_dict, r2, similarity

def train(feat, correct_feats, method='lstsq', tol=1e-4, l=1, max_iter=1000, epsilon=None, delta=1e-6, plot=False, normalize=False, clip_sd=None):
    """
    Train linear model for power usage

    :param feat: featureized data as pandas DataFrame
    :param method: type of linear model ('lstsq' or 'lasso')
    :return: model, coefficient dictionary, r2 score, convergence trace
    """
    type_fw = False
    if method == 'lstsq':
        model = LinearRegression()
    elif method == 'lasso':
        model = Lasso(alpha=l, max_iter=max_iter, tol=tol, fit_intercept=True)
    elif method == "fw-lasso":
        type_fw = True
    elif method == 'fw-lasso-exp':
        type_fw = True
    elif method == 'fw-lasso-lap':
        type_fw = True
    else:
        raise ValueError('method must be "lstsq" or "lasso"')

    if type_fw:
        #print("training frank wolfe model")
        #print(X_train.to_numpy().shape, y_train.to_numpy().shape)
        X, y = feat.drop(y_name, axis=1), feat[y_name]
        ones_column = np.ones((X.shape[0], 1))
        X_train, X_test, y_train, y_test = train_test_split(np.hstack((ones_column, X)), y, test_size=0.85, random_state=1)

        should_trace = True if plot else False
        if method == 'fw-lasso-lap':
            model = FWLasso.LaplaceNoise(X_train, y_train, l=l, delta=delta, epsilon=epsilon, K=max_iter, trace=should_trace, normalize=normalize, clip_sd=clip_sd)
        elif method == "fw-lasso":
            model = FWLasso.FW_NonPrivate(X_train, y_train, l=l, tol=tol, K=max_iter, normalize=normalize, clip_sd=clip_sd, trace=should_trace)
        else:
            model = FWLasso.ExponentialMechanism(X_train, y_train, l=l, delta=delta, epsilon=epsilon, K=max_iter, normalize=normalize, clip_sd=clip_sd, trace=should_trace)
        print(f'Train MSE fw: {mean_squared_error(y_train, X_train @ model.get("model")):.2f} ({100*sum(model.get("model")>0)/model.get("model").shape[0]:.1f}% sparse)')
        y_pred = X_test @ model.get("model")
        mse = mean_squared_error(y_test, y_pred)
        # print(f'Test MSE {method}: {mse:.2f} ({100*sum(model>0)/model.shape[0]:.1f}% sparse)')
        # Coefficient dictionary with feature name
        coef_dict = dict(zip(np.append(["Intercept"],X.columns), model.get("model")))
    else:
        X, y = feat.drop(y_name, axis=1), feat[y_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        # SciKit Model fitting
        model.fit(X_train, y_train)
        # MSE calculation
        y_pred = model.predict(X_test)
        print(f'Train MSE sklearn: {mean_squared_error(y_train, model.predict(X_train)):.2f} ({100*sum(model.coef_>0)/model.coef_.shape[0]:.1f}% sparse)')
        mse = mean_squared_error(y_test, y_pred)
        # print(f'Test MSE sklearn: {mse:.2f} ({100*sum(model.coef_>0)/model.coef_.shape[0]:.1f}% sparse)')
        # Coefficient dictionary with feature name
        coef_dict = dict(zip(X.columns, model.coef_))
        coef_dict["Intercept"] = model.intercept_

    r2 = r2_score(y_test, y_pred, force_finite=False)
    similarity = jaccard_similarity(correct_feats, [k for k, v in coef_dict.items() if v > 0])
    
    if plot:
        trace = model.get("plot")
        plt.clf()
        plt.plot(range(len(trace)), trace / max(trace), color="#00C7FD", lw=2)
        plt.yscale('log')
        plt.xlabel('Number of iterations')
        plt.ylabel('Delta f')
        plt.title(f'{method} Convergence')
        # plt.xlim()
        plt.grid()
        plt.tight_layout()
        plt.savefig(plot)
    return mse, coef_dict, r2, similarity
