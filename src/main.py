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

# Forward pass sanity check
sample_X = X_train[:32]

output = model(sample_X)

print("Input shape:", sample_X.shape)
print("Output shape:", output.shape)