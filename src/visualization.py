import matplotlib.pyplot as plt
import numpy as np

def inverse_transform_with_scaler(predictions, actuals, scaler):
    """
    스케일링 해제 과정에서 Close뿐만 아니라 Volume도 포함하여 처리.

    Args:
        predictions: 예측된 값
        actuals: 실제 값
        scaler: MinMaxScaler 객체

    Returns:
        predictions_inversed, actuals_inversed: 스케일링 해제된 값
    """
    # 더미 데이터를 생성하여 'Close'와 'Volume' 모두 포함
    dummy_data_predictions = np.zeros((predictions.shape[0], scaler.min_.shape[0]))
    dummy_data_predictions[:, 0] = predictions.flatten()  # Close 위치에 예측값 삽입

    dummy_data_actuals = np.zeros((actuals.shape[0], scaler.min_.shape[0]))
    dummy_data_actuals[:, 0] = actuals.flatten()  # Close 위치에 실제값 삽입

    # 전체 데이터를 스케일링 해제 후 'Close'만 추출
    predictions_inversed = scaler.inverse_transform(dummy_data_predictions)[:, 0]
    actuals_inversed = scaler.inverse_transform(dummy_data_actuals)[:, 0]

    return predictions_inversed, actuals_inversed

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

    # 스케일링 해제
    predictions_inversed, actuals_inversed = inverse_transform_with_scaler(predictions, actuals, scaler)

    # 테스트 데이터가 시작하는 인덱스
    test_start_idx = len(data) - len(predictions_inversed)

    print(f"Original data range: {data['Close'].min()} to {data['Close'].max()}")
    print(f"Predictions range after inverse transform: {predictions_inversed.min()} to {predictions_inversed.max()}")
    print(f"Actuals range after inverse transform: {actuals_inversed.min()} to {actuals_inversed.max()}")

    # 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"], label="Original Data", alpha=0.7)
    plt.plot(data.index[test_start_idx:], predictions_inversed, label="Predictions", color="red", linestyle="--")
    plt.plot(data.index[test_start_idx:], actuals_inversed, label="Actuals", color="green", linestyle="--")
    plt.title("Tesla Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
