import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

def calculate_annualized_volatility(symbol, start_date, end_date):
    try:
        df = yf.download(symbol + ".NS", start=start_date, end=end_date, progress=False)
        if df.empty:
            print(f"No data for {symbol}")
            return None
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        hv = np.std(df['log_return'].dropna()) * np.sqrt(252)
        return hv
    except Exception as e:
        print(f"Error with {symbol}: {e}")
        return None

def filter_stable_stocks(stock_list, hv_threshold=0.35):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)  # 5 years

    hv_results = {}
    for stock in stock_list:
        hv = calculate_annualized_volatility(stock, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if hv is not None:
            hv_results[stock] = hv

    # Sort stocks by volatility
    sorted_stocks = sorted(hv_results.items(), key=lambda x: x[1])

    # Filter stocks below threshold
    stable_stocks = [stock for stock, vol in sorted_stocks if vol < hv_threshold]

    print("\nAll Stocks and Their Annualized Volatility (Past 5 Years):")
    for stock, vol in sorted_stocks:
        print(f"{stock}: {vol:.2f}")

    print("\n✔️ Stable Stocks Suitable for 5-Year Investment (HV < {:.2f}):".format(hv_threshold))
    print(stable_stocks)

    return stable_stocks, hv_results
stock_list = ["TCS", "INFY", "RELIANCE", "ZOMATO", "PAYTM", "ADANIENT", "HDFC", "ITC", "DMART"]
stable_stocks, volatility_data = filter_stable_stocks(stock_list, hv_threshold=0.35)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_stock_data(symbol, start='2010-01-01', end='2024-12-31'):
    df = yf.download(symbol + ".NS", start=start, end=end)
    df = df[['Close']].dropna()
    return df

def prepare_lstm_data(df, time_steps=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(time_steps, len(scaled)):
        X.append(scaled[i - time_steps:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler

from keras import Sequential
from keras.src.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Predict next price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_price(model, df, scaler, time_steps=60, days_ahead=1260):  # 5 years = ~1260 trading days
    last_sequence = df[-time_steps:].values
    scaled_seq = scaler.transform(last_sequence)
    predicted = []

    input_seq = scaled_seq.copy()
    for _ in range(days_ahead):
        X_input = np.array(input_seq[-time_steps:]).reshape(1, time_steps, 1)
        next_price = model.predict(X_input, verbose=0)
        predicted.append(next_price[0][0])
        input_seq = np.append(input_seq, [[next_price]], axis=0)

    predicted_prices = scaler.inverse_transform(np.array(predicted).reshape(-1, 1))
    return predicted_prices[-1][0] 

def compare_returns(stable_stocks):
    returns = {}
    for stock in stable_stocks:
        df = get_stock_data(stock)
        X, y, scaler = prepare_lstm_data(df)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = build_lstm_model((X.shape[1], 1))
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)

        future_price = predict_future_price(model, df[['Close']], scaler)
        current_price = df.iloc[-1]['Close']
        returns[stock] = (future_price - current_price) / current_price

    return sorted(returns.items(), key=lambda x: x[1], reverse=True)
ans=compare_returns(stable_stocks)
print("\nStocks Sorted by Expected Returns (Next 5 Years):")
for stock, ret in ans:
    print(f"{stock}: {ret:.2%}")