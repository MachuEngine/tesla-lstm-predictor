import numpy as np
import yfinance as yf
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

def dataset():
    # TSLA 주가 데이터 다운로드 (예: 최근 5년간의 일일 주가)
    ticker = "TSLA"
    data = yf.download(ticker, period="5y", interval="1d")
    data = data[['Close', 'Volume']]  # 2차원 데이터로 유지

    # Train-test split
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size].copy()  # 2차원 데이터
    test_data = data[train_size:].copy()

    # Close와 Volume을 별도로 스케일링
    def scale_features(train_data, test_data):
        # Close와 Volume에 대해 각각 다른 스케일러 사용
        """
        [scale_features 함수 실행 전]
        Date	    Close	Volume
        2020-01-01	100.0	1,000,000
        2020-01-02	102.0	1,200,000
        2020-01-03	98.0	950,000

        [scale_features 함수 실행 후]
        Date	    Close	Volume	    Close_scaled	Volume_scaled
        2020-01-01	100.0	1,000,000	0.5	            0.3333
        2020-01-02	102.0	1,200,000	1.0	            1.0
        2020-01-03	98.0	950,000	    0.0	            0.0
        """
        close_scaler = MinMaxScaler(feature_range=(0, 1))
        volume_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Close 피처에 대한 스케일링
        train_data["Close_scaled"] = close_scaler.fit_transform(train_data["Close"].values.reshape(-1, 1))
        test_data["Close_scaled"] = close_scaler.transform(test_data["Close"].values.reshape(-1, 1))
        
        # Volume 피처에 대한 스케일링
        train_data["Volume_scaled"] = volume_scaler.fit_transform(train_data["Volume"].values.reshape(-1, 1))
        test_data["Volume_scaled"] = volume_scaler.transform(test_data["Volume"].values.reshape(-1, 1))

        return train_data, test_data, close_scaler, volume_scaler

    train_data, test_data, close_scaler, volume_scaler = scale_features(train_data, test_data)

    # 시퀀스 데이터 준비 함수
    def create_sequences(dataset, time_step=60):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            # Close_scaled와 Volume_scaled를 함께 사용
            X.append(dataset[["Close_scaled", "Volume_scaled"]].iloc[i:i+time_step].values)
            y.append(dataset["Close_scaled"].iloc[i+time_step])  # Close_scaled를 예측 대상으로 설정
        return np.array(X), np.array(y)

    time_step = 60
    X_train, y_train = create_sequences(train_data, time_step)
    X_test, y_test = create_sequences(test_data, time_step)

    # Tensor 변환 및 차원 추가
    X_train = torch.tensor(X_train, dtype=torch.float32)  # (samples, time_step, features)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)  # Match output size with input
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    print(f"y_train min: {y_train.min()}, max: {y_train.max()}")
    print(f"y_test min: {y_test.min()}, max: {y_test.max()}")

    # TensorDataset 및 DataLoader 생성
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader, close_scaler, volume_scaler, data
