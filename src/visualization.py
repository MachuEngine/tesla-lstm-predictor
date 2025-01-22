import matplotlib.pyplot as plt

def plot_predictions(predictions, actuals, scaler, data):
    """
    예측값과 실제값을 시각화하는 함수.

    Args:
        predictions (ndarray): 예측된 값 (Test 데이터셋 기반).
        actuals (ndarray): 실제 값 (Test 데이터셋 기반).
        scaler (MinMaxScaler): 데이터 스케일링에 사용된 스케일러.
        data (DataFrame): 원본 데이터셋.
    """
    print(f"Predictions before inverse transform: {predictions[:10]}")


    # 예측값과 실제값 스케일 복원
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    actuals = scaler.inverse_transform(actuals.reshape(-1, 1))

    # 테스트 데이터가 시작하는 인덱스
    test_start_idx = len(data) - len(predictions)

    print(f"Original data range: {data['Close'].min()} to {data['Close'].max()}")
    print(f"Predictions range after inverse transform: {predictions.min()} to {predictions.max()}")
    print(f"Actuals range after inverse transform: {actuals.min()} to {actuals.max()}")


    # 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"], label="Original Data", alpha=0.7)
    plt.plot(data.index[test_start_idx:], predictions, label="Predictions", color="red", linestyle="--")
    plt.plot(data.index[test_start_idx:], actuals, label="Actuals", color="green", linestyle="--")
    plt.title("Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

