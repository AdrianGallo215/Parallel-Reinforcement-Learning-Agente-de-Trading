import yfinance as yf
import pandas_ta as ta
import pandas as pd

def download_and_clean_data(ticker, start, end):
    raw_data = yf.download(ticker, start=start, end=end)
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.droplevel('Ticker')
    df = raw_data[['Volume', 'Close']].copy()
    df['close_norm'] = df['Close'].pct_change()
    df['volume_norm'] = df['Volume'].pct_change()
    df = get_indicators(df)
    df.dropna(inplace=True) 
    return df

def get_train_test(df, test_ratio = 0.2):
    split_idx = int(len(df) * (1 - test_ratio))
    train_data = df.iloc[:split_idx].copy()
    test_data = df.iloc[split_idx:].copy()
    return train_data, test_data

def standardize_features(train_data, test_data, cols=("RSI", "macd_signal")):
    """Z-score usando media/std del TRAIN, aplicado igual a train y test.
    No usar .std()/.mean() del test: eso filtraría información del futuro."""
    train_data = train_data.copy()
    test_data = test_data.copy()
    for col in cols:
        mu = train_data[col].mean()
        sigma = train_data[col].std()
        train_data[col] = (train_data[col] - mu) / sigma
        test_data[col] = (test_data[col] - mu) / sigma
    return train_data, test_data

def get_indicators(df):
    close = df['Close'].squeeze()
    df['RSI'] = ta.rsi(close, length=14)
    macd_df = ta.macd(close, fast = 12, slow = 26, signal = 9)
    if macd_df is not None and 'MACDs_12_26_9' in macd_df.columns:
        df['macd_signal'] = macd_df['MACDs_12_26_9'].values
    else:
        # Fallback: compute MACD signal manually
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        df['macd_signal'] = macd_line.ewm(span=9, adjust=False).mean().values
    return df

