import torch
import torch.nn as nn


def evaluate_model(model, X_test, y_test):

    criterion = nn.MSELoss()

    model.eval()

    with torch.no_grad():

        predictions = model(X_test)

        loss = criterion(predictions, y_test)

    print(f"Test Loss: {loss.item():.6f}")

    return predictions