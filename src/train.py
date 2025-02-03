import numpy as np
import torch
import torch.nn as nn
from src.model import LSTMModel

def train_lstm_model(model, train_loader, config):
    # 모델 초기화
    # 모델을 초기화하게 되면서 평가 시 학습되지 않은 모델로 평가하는 문제가 발생됨 (이를 방지하기 위해 모델을 트레이닝 단계에서 초기화하지 않도록 주석 처리)
    # model = LSTMModel(config["train"]["input_size"], config["train"]["hidden_size"], config["train"]["num_layers"], config["train"]["output_size"], 0.2)

    # 장치 설정 (GPU 사용 가능 시 GPU 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # MSE 대신 Huber Loss 사용
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    num_epochs = config["train"]["num_epochs"]
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        scheduler.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")