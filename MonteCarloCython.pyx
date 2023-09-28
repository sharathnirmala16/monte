# cython: language_level=3
# cython: boundscheck=False

cimport cython
cimport numpy as np
import numpy as np

cpdef np.ndarray[double] simulate(np.ndarray[double] prices, int candles, int iterations):
    cdef double mu, var, stdev, drift
    cdef np.ndarray[double] log_returns, pct_change
    cdef np.ndarray[double, ndim=2] Z = np.empty((candles, iterations), dtype=np.double)
    cdef np.ndarray[double, ndim=2] daily_returns = np.empty((candles, iterations), dtype=np.double)
    cdef np.ndarray[double, ndim=2] price_paths = np.empty((candles, iterations), dtype=np.double)

    pct_change = np.zeros_like(prices)
    pct_change[0] = 0
    pct_change[1:] = prices[1:] / prices[:-1] - 1
    log_returns = np.log(1 + pct_change)

    mu = log_returns.mean()
    var = log_returns.var()
    stdev = log_returns.std()
    drift = mu - (0.5 * var * (prices.shape[0] - 2) / prices.shape[0])

    Z = np.random.laplace(size=(candles, iterations))
    daily_returns = np.exp(drift + stdev * Z)

    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = prices[-1]

    for t in range(1, candles):
        price_paths[t] = price_paths[t - 1] * daily_returns[t]

    return price_paths