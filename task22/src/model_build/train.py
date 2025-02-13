from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import model_build.frankWolfeLASSO as fw # how to do this relatively?

y_name = 'power_mean'

def train(feat, type='lstsq', tol=1e-4, l=1, max_iter=1000, epsilon=None, delta=1e-6):
    """
    Train linear model for power usage

    :param feat: featureized data as pandas DataFrame
    :param type: type of linear model ('lstsq' or 'lasso')
    :return: model, coefficient dictionary, r2 score
    """
    type_fw = False
    if type == 'lstsq':
        model = LinearRegression()
    elif type == 'lasso':
        model = Lasso(alpha=l, max_iter=max_iter, tol=tol)
    elif type == 'fw-lasso':
        type_fw = True
    else:
        raise ValueError('type must be "lstsq" or "lasso"')

    # Data split
    X, y = feat.drop(y_name, axis=1), feat[y_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


    if type_fw:
        #print("training frank wolfe model")
        #print(X_train.to_numpy().shape, y_train.to_numpy().shape)
        model = fw.frankWolfeLASSO(X_train.to_numpy(), y_train.to_numpy(), l=l, tol=tol, K=max_iter, delta=delta, epsilon=epsilon) #l is alpha , diffx, k 
        # k = The number of iterations made
        # diffx = The difference between the last two iterations
        y_pred = X_test.dot(model)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Test MSE FW-Alg: {mse:.2f} ({100*sum(model>0)/model.shape[0]:.1f}% sparse)')
        # Coefficient dictionary with feature name
        coef_dict = dict(zip(X.columns, model))
    else:
        # SciKit Model fitting
        model.fit(X_train, y_train)
        # MSE calculation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Test MSE sklearn: {mse:.2f} ({100*sum(model.coef_>0)/model.coef_.shape[0]:.1f}% sparse)')
        # Coefficient dictionary with feature name
        coef_dict = dict(zip(X.columns, model.coef_))

    r2 = r2_score(y_test, y_pred)
    
    return mse, coef_dict, r2
