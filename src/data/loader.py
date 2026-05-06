import yfinance as yf

def fetch_data(symbol="BTC-USD", start="2018-01-01"):
    df = yf.download(symbol, start=start)
    df = df[['Close']]
    df.dropna(inplace=True)
    return df