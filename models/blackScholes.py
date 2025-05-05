import yfinance as yf
import numpy as np

def get_stock_volatility(ticker, period="1y"):
    data = yf.download(ticker, period=period)
    data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
    volatility = np.std(data['returns'].dropna()) * np.sqrt(252)  # Annualized volatility
    return volatility
from scipy.stats import norm
import numpy as np

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price
ticker = "AAPL"
current_price = yf.Ticker(ticker).history(period="1d")['Close'][0]
volatility = get_stock_volatility(ticker)
risk_free_rate = 0.05  # 5%
time_to_goal = 2  # 2 years
strike_price = current_price  # Assume ATM (at-the-money)

call_option_price = black_scholes_call(current_price, strike_price, time_to_goal, risk_free_rate, volatility)

print(f"Estimated Call Option Price for {ticker}: ${call_option_price:.2f}")