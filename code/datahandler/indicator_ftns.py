import ta
import numpy as np


def add_basic_indicators(df):
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['return_5'] = df['close'].pct_change(periods=5)
    df['return_10'] = df['close'].pct_change(periods=10)
    df['volume_change'] = df['vol'].pct_change()
    return df

def add_trend_indicators(df):
    df['ema_5'] = df['close'].ewm(span=5).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_cross'] = df['ema_5'] - df['ema_20']
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
    df['sar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
    return df

def add_momentum_indicators(df):
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df['%K'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df['%D'] = df['%K'].rolling(window=3).mean()
    df['roc'] = df['close'].pct_change(periods=12)
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    return df

def add_volume_indicators(df):
    direction = np.sign(df['close'].diff())
    df['obv'] = (direction * df['vol']).fillna(0).cumsum()
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-9)
    df['ad_line'] = (clv * df['vol']).cumsum()
    return df

def add_volatility_indicators(df):
    if len(df) < 20:
        # NaN 채널을 생성하여 column shape 일관성 유지
        df['bb_upper'] = np.nan
        df['bb_lower'] = np.nan
        df['bb_width'] = np.nan
        df['atr'] = np.nan
        df['gap_size'] = df['open'] - df['prevClose']
        return df
    
    ma = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = ma + (2 * std)
    df['bb_lower'] = ma - (2 * std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / ma
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['gap_size'] = df['open'] - df['prevClose']
    return df