import torch

from src.data.loader import fetch_data
from src.data.preprocess import compute_returns, scale_data
from src.data.features import create_sequences
from src.models.lstm import LSTMModel


# Data ingestion
df = fetch_data()

# Convert prices to log returns
df = compute_returns(df)

# Extract returns column
data = df[['returns']].values

# Scale data
scaled_data, scaler = scale_data(data)

# Create sequences
X, y = create_sequences(scaled_data)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Time-aware train/test split
train_size = int(0.8 * len(X))

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Initialize model
model = LSTMModel()

# Forward pass sanity check
sample_X = X_train[:32]

output = model(sample_X)

print("Input shape:", sample_X.shape)
print("Output shape:", output.shape)