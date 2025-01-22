import numpy as np
import torch
import torch.nn as nn
from src.model import LSTMModel

def train_lstm_model(model, train_loader, config):
    # 모델 초기화
    model = LSTMModel(config["train"]["input_size"], config["train"]["hidden_size"], config["train"]["num_layers"], config["train"]["output_size"])

    # 장치 설정 (GPU 사용 가능 시 GPU 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # MSE 대신 Huber Loss 사용
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

    num_epochs = config["train"]["num_epochs"]

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # 에포크별 학습 손실 출력
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(train_losses):.6f}")
