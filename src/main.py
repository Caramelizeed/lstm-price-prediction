import torch
import numpy as np
from src.data.loader import fetch_data
from src.data.preprocess import compute_returns, scale_data
from src.data.features import create_sequences
from src.models.lstm import LSTMModel

from src.models.train import train_model
from src.evaluation.metrics import evaluate_model

# Data ingestion
df = fetch_data()

# Convert prices to log returns
df = compute_returns(df)

# Extract returns column
data = df[['returns']].values

# Split BEFORE scaling
train_size = int(0.8 * len(data))

train_data = data[:train_size]
test_data = data[train_size:]

# Fit scaler only on training data
scaled_train, scaler = scale_data(train_data)

# Transform test data using same scaler
scaled_test = scaler.transform(test_data)

# Create sequences separately
X_train, y_train = create_sequences(scaled_train)
X_test, y_test = create_sequences(scaled_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Initialize model
model = LSTMModel()

#train model 
train_model(model, X_train, y_train)
# Evaluate model
predictions, test_loss = evaluate_model(model, X_test, y_test)

predictions = predictions.numpy()
actuals = y_test.numpy()

print("\nFirst 10 Predictions vs Actuals:\n")

for i in range(10):
    print(
        f"Pred: {predictions[i][0]:.6f} | "
        f"Actual: {actuals[i][0]:.6f}"
    )
pred_signs = np.sign(predictions)
actual_signs = np.sign(actuals)

directional_accuracy = np.mean(pred_signs == actual_signs)

print(f"\nDirectional Accuracy: {directional_accuracy:.4f}")

# Forward pass sanity check
sample_X = X_train[:32]

output = model(sample_X)



print("Input shape:", sample_X.shape)
print("Output shape:", output.shape)