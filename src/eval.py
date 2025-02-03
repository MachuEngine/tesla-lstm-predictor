import numpy as np
import torch

def evaluate(model, test_loader, close_scaler):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # 역변환
    predictions_inversed = close_scaler.inverse_transform(predictions)
    actuals_inversed = close_scaler.inverse_transform(actuals)

    print(f"Predictions before inverse transform: {predictions[:10]}")
    print(f"Predictions range after inverse transform: {predictions_inversed.min()} to {predictions_inversed.max()}")
    print(f"Actuals range after inverse transform: {actuals_inversed.min()} to {actuals_inversed.max()}")

    return predictions_inversed, actuals_inversed
