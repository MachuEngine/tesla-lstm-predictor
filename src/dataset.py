import numpy as np
import yfinance as yf
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

def dataset():
    # TSLA 주가 데이터 다운로드 (예: 최근 10년간의 일일 주가)
    ticker = "TSLA"
    data = yf.download(ticker, period="10y", interval="1d")
    data = data[['Close', 'Volume', 'Open', 'High', 'Low']]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train-test split
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size].copy()
    test_data = data[train_size:].copy()

    # 학습 데이터에 대해 스케일링
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    volume_scaler = MinMaxScaler(feature_range=(0, 1))
    open_scaler = MinMaxScaler(feature_range=(0, 1))
    high_scaler = MinMaxScaler(feature_range=(0, 1))
    low_scaler = MinMaxScaler(feature_range=(0, 1))

    train_data["Close_scaled"] = close_scaler.fit_transform(train_data["Close"].values.reshape(-1, 1))
    train_data["Volume_scaled"] = volume_scaler.fit_transform(train_data["Volume"].values.reshape(-1, 1))
    train_data["Open_scaled"] = open_scaler.fit_transform(train_data["Open"].values.reshape(-1, 1))
    train_data["High_scaled"] = high_scaler.fit_transform(train_data["High"].values.reshape(-1, 1))
    train_data["Low_scaled"] = low_scaler.fit_transform(train_data["Low"].values.reshape(-1, 1))

    # 테스트 데이터는 학습 데이터의 스케일러를 사용하여 변환
    test_data["Close_scaled"] = close_scaler.transform(test_data["Close"].values.reshape(-1, 1))
    test_data["Volume_scaled"] = volume_scaler.transform(test_data["Volume"].values.reshape(-1, 1))
    test_data["Open_scaled"] = open_scaler.transform(test_data["Open"].values.reshape(-1, 1))
    test_data["High_scaled"] = high_scaler.transform(test_data["High"].values.reshape(-1, 1))
    test_data["Low_scaled"] = low_scaler.transform(test_data["Low"].values.reshape(-1, 1))

    # 시퀀스 데이터 준비 함수
    def create_sequences(dataset, time_step=30):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[["Close_scaled", "Volume_scaled", "Open_scaled", "High_scaled", "Low_scaled"]].iloc[i:i+time_step].values)
            y.append(dataset["Close_scaled"].iloc[i+time_step])
        return np.array(X), np.array(y)

    time_step = 30
    X_train, y_train = create_sequences(train_data, time_step)
    X_test, y_test = create_sequences(test_data, time_step)

    # Tensor 변환 및 차원 추가
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1).to(device)

    print(f"y_train min: {y_train.min()}, max: {y_train.max()}")
    print(f"y_test min: {y_test.min()}, max: {y_test.max()}")

    # TensorDataset 및 DataLoader 생성
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader, close_scaler, data