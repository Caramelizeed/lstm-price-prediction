import torch
from src.data.loader import fetch_data
from src.data.preprocess import compute_returns, scale_data
from src.data.features import create_sequences
from src.models.lstm import LSTMModel

df = fetch_data()
df = compute_returns(df)

data = df[['returns']].values
scaled_data, scaler = scale_data(data)

X, y = create_sequences(scaled_data)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

model = LSTMModel()

# take small batch
sample_X = X[:32]

output = model(sample_X)

print("Input shape:", sample_X.shape)
print("Output shape:", output.shape)