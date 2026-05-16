import yfinance as yf
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def get_stock_data(symbol, start_date, end_date):
    print(f"Baixando dados para {symbol}...")
    df = yf.download(symbol, start=start_date, end=end_date)
    return df[['Close']].values

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

def prepare_data(symbol, start_date, end_date, seq_length=60):
    data = get_stock_data(symbol, start_date, end_date)
    
    # Voltamos a usar apenas um scaler simples
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = create_sequences(scaled_data, seq_length)
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    return X, y, scaler