import numpy as np
import torch

def evaluate(model, test_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            # Ensure batch_X and batch_y are tensors
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())

    return np.concatenate(predictions), np.concatenate(actuals)
