import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(predictions, actuals, data):
    print(f"Predictions before inverse transform: {predictions[:10]}")


    # 테스트 데이터가 시작하는 인덱스
    test_start_idx = len(data) - len(predictions)

    print(f"Original data range: {data['Close'].min()} to {data['Close'].max()}")
    print(f"Predictions range after inverse transform: {predictions.min()} to {predictions.max()}")
    print(f"Actuals range after inverse transform: {actuals.min()} to {actuals.max()}")

    # 시각화
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data["Close"], label="Original Data", alpha=0.7)
    plt.plot(data.index[test_start_idx:], predictions, label="Predictions", color="red", linestyle="--")
    plt.plot(data.index[test_start_idx:], actuals, label="Actuals", color="green", linestyle="--")
    plt.title("Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
