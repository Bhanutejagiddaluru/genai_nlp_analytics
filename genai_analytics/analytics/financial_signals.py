import pandas as pd
import numpy as np

def _synthetic_prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    price = 100 * np.exp(np.cumsum(returns))
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq='B')
    return pd.DataFrame({'close': price}, index=idx)

def rsi(series: pd.Series, window: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_signals(symbol: str = 'AAPL', n: int = 200):
    # For demo we use synthetic prices; plug in real data in production.
    df = _synthetic_prices(n=n, seed=hash(symbol) % 2**32)
    df['rsi'] = rsi(df['close'])
    macd_line, signal_line, hist = macd(df['close'])
    df['macd'] = macd_line; df['signal'] = signal_line; df['hist'] = hist
    feats = {'latest_close': float(df['close'].iloc[-1]), 'latest_rsi': float(df['rsi'].iloc[-1])}
    return df, feats
