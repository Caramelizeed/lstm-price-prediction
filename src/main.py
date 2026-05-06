from src.data.loader import fetch_data

#data ingestion

df = fetch_data()
print(df.head())
print(df.shape)