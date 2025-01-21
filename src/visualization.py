import matplotlib.pyplot as plt

def plot_predictions(data, predictions, actuals, scaler):
    # 정규화 해제 (원래 값 복원)
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)

    # 실제 날짜 인덱스 가져오기 (테스트 데이터 부분)
    test_data = data.iloc[-len(actuals):]  # 실제 test 기간에 해당하는 데이터

    # 시각화
    plt.figure(figsize=(12,6))
    plt.plot(test_data.index, actuals, label='Test Actual')
    plt.plot(test_data.index, predictions, label='Test Prediction')
    plt.xlabel('Date')
    plt.ylabel('Tesla Close Price')
    plt.legend()
    plt.show()