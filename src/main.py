from src.data.loader import fetch_data
from src.data.preprocess import compute_returns

df = fetch_data()
df = compute_returns(df)

print(df.head())
print(df.shape)