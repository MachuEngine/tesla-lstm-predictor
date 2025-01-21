import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

def dataset():
    # TSLA 주가 데이터 다운로드 (예: 최근 5년간의 일일 주가)
    ticker = "TSLA"
    data = yf.download(ticker, period="5y", interval="1d")
    data = data[['Close']]  # 종가만 사용

    # 데이터 정규화 (0과 1 사이로 변환)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # 시퀀스 데이터 준비 함수
    def create_sequences(dataset, time_step=60):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X, y = create_sequences(scaled_data, time_step)

    # PyTorch 텐서로 변환 및 차원 재조정
    X = torch.from_numpy(X).float().unsqueeze(-1)  # shape: (samples, timesteps, features)
    y = torch.from_numpy(y).float().unsqueeze(-1)  # shape: (samples, 1)


    dataset = TensorDataset(X, y)

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader, scaler