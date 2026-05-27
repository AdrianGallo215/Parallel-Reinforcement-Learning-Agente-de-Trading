import yfinance as yf
import pandas_ta as ta

def download_and_clean_data(ticker, start, end):
    raw_data = yf.download(ticker, start=start, end=end)
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

def get_indicators(df):
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd_df = ta.macd(df['Close'], fast = 12, slow = 26, signal = 9)
    df['macd_signal'] = macd_df['MACDs_12_26_9']
    return df
