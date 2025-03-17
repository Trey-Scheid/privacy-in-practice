"""
File: train_redshift.py
Author: Cheon
Date: 
Description: train regression model of choice
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


y_name = 'power_mean'

def train(feat, type='lstsq'):
    """
    Train linear model for power usage

    :param feat: featureized data as pandas DataFrame
    :param type: type of linear model ('lstsq' or 'lasso')
    :return: model, coefficient dictionary, r2 score
    """
    if type == 'lstsq':
        model = LinearRegression()
    elif type == 'lasso':
        model = Lasso()
    else:
        raise ValueError('type must be "lstsq" or "lasso"')

    # Data split
    X, y = feat.drop(y_name, axis=1), feat[y_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Model fitting
    model.fit(X_train, y_train)

    # MSE calculation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Test MSE: {mse:.2f}')

    # Coefficient dictionary with feature name
    coef_dict = dict(zip(X.columns, model.coef_))

    r2 = r2_score(y_test, y_pred)
    
    return model, coef_dict, r2