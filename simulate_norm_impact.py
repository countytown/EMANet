import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
import time

def generate_dataset():
    # generate data
    np.random.seed()
    X = np.random.rand(10000, 2)  # input feature
    noise = np.random.normal(0, 0.5, 10000)  # add noises
    y = 3 * X[:,0] + 5 * X[:,1] + noise  # target label
    return X, y

def normalize_data(X, norm_type):
    if norm_type == 'l2':
        X_normalized = normalize(X, norm='l2')
    elif norm_type == 'sum':
        X_normalized = X / np.sum(X, axis=1, keepdims=True)
    else:
        raise ValueError("Invalid norm type. Use 'l2' or 'sum'.")
    return X_normalized

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def simulate_models():
    # set random seed so every time has different data
    np.random.seed(int(time.time()))

    # generate data
    X, y = generate_dataset()

    # split to train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # init result dictionary
    results = {'l2_linear': None, 'sum_linear': None, 'l2_poly': None, 'sum_poly': None}

    # two kinds of normalization
    for norm_type in ['l2', 'sum']:
        for model_type in ['linear', 'poly']:
            # norm
            X_train_normalized = normalize_data(X_train, norm_type)
            X_test_normalized = normalize_data(X_test, norm_type)

            # poly, feature transform
            if model_type == 'poly':
                poly = PolynomialFeatures(degree=2)
                X_train_normalized = poly.fit_transform(X_train_normalized)
                X_test_normalized = poly.transform(X_test_normalized)

            # train and validate
            mse = train_and_evaluate_model(X_train_normalized, X_test_normalized, y_train, y_test)
            results[f'{norm_type}_{model_type}'] = mse

    return results

run_times = 4
# run simulation
for run in range(run_times):
    results = simulate_models()
    print('Run', run+1)
    for key, value in results.items():
        print(f'{key}: {value}')
