from sklearn.ensemble import RandomForestRegressor
import numpy as np

def make_sliding_window(series, window=20, horizon=1):
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i:i+window])
        y.append(series[i+window:i+window+horizon])
    return np.array(X), np.array(y).ravel()

def rf_forecast(series, window=20, horizon=1):
    X, y = make_sliding_window(series, window=window, horizon=horizon)
    model = RandomForestRegressor(n_estimators=50, random_state=0)
    model.fit(X, y)
    pred = model.predict([series[-window:]])
    return float(pred[0])
