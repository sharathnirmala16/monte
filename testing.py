import numpy as np
import pandas as pd
import yfinance as yf
from MonteCarlo import MonteCarlo

df: pd.DataFrame = yf.download(tickers=["TCS.NS"], period="5y", interval="1d")

mc = MonteCarlo(10000, 60)
paths = mc.simulate(df["Close"])
print(mc.expected_returns(paths))
