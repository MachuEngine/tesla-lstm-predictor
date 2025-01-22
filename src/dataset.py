import numpy as np
import yfinance as yf
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

def dataset():
    # TSLA 주가 데이터 다운로드 (예: 최근 5년간의 일일 주가)
    ticker = "TSLA"
    data = yf.download(ticker, period="5y", interval="1d")
    data = data[['Close']]  # 종가만 사용

    # Train-test split (먼저 나눔)
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # 데이터 정규화 (훈련 데이터에 대해서만 fit, 테스트 데이터는 훈련 데이터 기반으로 fit한 정규화 기반으로 transform)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = scaler.fit_transform(train_data)  # 훈련 데이터에 대해 fit_transform
    scaled_test_data = scaler.transform(test_data)       # 테스트 데이터에 대해 transform

    # 시퀀스 데이터 준비 함수
    def create_sequences(dataset, time_step=120):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 120
    X_train, y_train = create_sequences(scaled_train_data, time_step)
    X_test, y_test = create_sequences(scaled_test_data, time_step)

    # Tensor 변환 및 차원 추가
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # Add input_size dimension
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)  # Match output size with input
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)  # Add input_size dimension
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)  # Match output size with input

    print(f"y_train min: {y_train.min()}, max: {y_train.max()}")
    print(f"y_test min: {y_test.min()}, max: {y_test.max()}")

    # TensorDataset 및 DataLoader 생성
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader, scaler, data

