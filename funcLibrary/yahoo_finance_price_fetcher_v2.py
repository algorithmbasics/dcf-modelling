# file: notification_app.py
# add emoticons to logging messages for quick diagnosis
# Example usage: success: ✅, warning: ⚠️, error: ❌


import yfinance as yf

def yahoo_finance_price_fetcher_v2(symbol: str) -> float:
    return yf.Ticker(symbol).fast_info.get("lastPrice")
