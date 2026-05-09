import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data.loader import fetch_data
from src.data.preprocess import compute_returns, scale_data
from src.data.features import create_sequences, add_features

from src.models.lstm import LSTMModel
from src.models.train import train_model

from src.evaluation.metrics import evaluate_model


# -----------------------------
# Data ingestion
# -----------------------------
df = fetch_data()

# Convert prices to log returns
df = compute_returns(df)

# Add technical indicators
df = add_features(df)


# -----------------------------
# Feature / target separation
# -----------------------------
feature_columns = [
    'returns',
    'sma_10',
    'ema_10',
    'volatility_10',
    'rsi_14'
]

# Input features
X_data = df[feature_columns].values

# Binary direction target
y_data = (df['returns'] > 0).astype(int).values.reshape(-1, 1)


# -----------------------------
# Train / Test split
# -----------------------------
train_size = int(0.8 * len(X_data))

X_train_raw = X_data[:train_size]
X_test_raw = X_data[train_size:]

y_train_raw = y_data[:train_size]
y_test_raw = y_data[train_size:]


# -----------------------------
# Scale INPUT features only
# -----------------------------
scaled_train, scaler = scale_data(X_train_raw)

scaled_test = scaler.transform(X_test_raw)


# -----------------------------
# Create sequences
# -----------------------------
X_train, y_train = create_sequences(
    scaled_train,
    y_train_raw
)

X_test, y_test = create_sequences(
    scaled_test,
    y_test_raw
)


# -----------------------------
# Convert to tensors
# -----------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# -----------------------------
# Initialize model
# -----------------------------
model = LSTMModel()


# -----------------------------
# Train model
# -----------------------------
train_model(model, X_train, y_train)


# -----------------------------
# Evaluate model
# -----------------------------
predictions, test_loss = evaluate_model(
    model,
    X_test,
    y_test
)

predictions = predictions.numpy()
actuals = y_test.numpy()


# -----------------------------
# Display predictions
# -----------------------------
print("\nFirst 10 Predictions vs Actuals:\n")

for i in range(10):

    print(
        f"Pred: {predictions[i][0]:.6f} | "
        f"Actual: {actuals[i][0]:.6f}"
    )


# -----------------------------
# Directional accuracy
# -----------------------------
pred_signs = np.sign(predictions)
actual_signs = np.sign(actuals)

directional_accuracy = np.mean(
    pred_signs == actual_signs
)

print(f"\nDirectional Accuracy: {directional_accuracy:.4f}")

#---------------------------
#plit predictions vs actuals
#---------------------------

plt.figure(figsize=(12, 6))
plt.plot(
    actuals[:200],
    label='Actual'
)
plt.plot(
    predictions[:200],
    label='Predicted'
)
plt.title('Predicted vs Actual Returns (First 200 Points)')
plt.xlabel('Time Step')
plt.ylabel('Log Return')
plt.legend()
plt.show()
