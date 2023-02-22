# -*- coding: utf-8 -*-
"""
Trend following example.

@author: Katie He
"""
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

###############################################################################
# Moving average functions
###############################################################################

def sma(x, n_lags):
    """
    Simple moving average

    Parameters
    ----------
    x : Numpy array
        Input time series.
    n_lags : int
        Length of moving average window.

    Returns
    -------
    y : Numpy array
        Output moving average series.

    """
    # Filter
    v = np.ones((n_lags,))/n_lags
    
    # Moving average
    y = np.convolve(x, v, mode='valid')
    
    # Align
    y = np.concatenate((np.zeros((n_lags-1,))*np.nan, y))
    return y

def ema(x, halflife):
    """
    Exponential moving average. More details here: 
        https://en.wikipedia.org/wiki/Exponential_smoothing

    Parameters
    ----------
    x : Numpy array
        Input time series.
    halflife : int or float
        Filter half-life.

    Returns
    -------
    y : Numpy array
        Output moving average series.

    """
    # Calculate smoothing factor
    alpha = 1-(1/2)**(1/halflife)
    
    # Initialize
    y = np.zeros_like(x)
    y[0] = x[0]
    
    # Moving average
    for t in range(1, len(x)):
        y[t] = alpha*x[t]+(1-alpha)*y[t-1]
        
    return y

###############################################################################
# SMA trend following
###############################################################################

# Download S&P data from Yahoo
data = yf.download('^GSPC', start='2017-02-01', end='2023-02-24')
settle = data['Close']

# Lag
n_fast = 50
n_slow = 100

# Moving averages
sma_fast = sma(settle, n_fast)
sma_slow = sma(settle, n_slow)

# Plot
plt.figure(figsize=(10,4))
plt.plot(data.index, settle, label='Daily Settle')
plt.plot(data.index, sma_fast, label='%s-day MA'%n_fast)
plt.plot(data.index, sma_slow, label='%s-day MA'%n_slow)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('S&P Settle and Simple Moving Averages')

###############################################################################
# EMA trend following
###############################################################################

# Half-life
hl_fast = 12.5
hl_slow = 25

# Moving averages
ema_fast = ema(settle, hl_fast)
ema_slow = ema(settle, hl_slow)

# Plot
plt.figure(figsize=(10,4))
plt.plot(data.index, settle, label='Daily Settle')
plt.plot(data.index, ema_fast, label='Half-life: %s'%hl_fast)
plt.plot(data.index, ema_slow, label='Half-life: %s'%hl_slow)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('S&P Settle and Exponential Moving Averages')
