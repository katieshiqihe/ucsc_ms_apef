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
sp_data = yf.download('^GSPC', start='2019-02-01', end='2024-02-09')
sp_settle = sp_data['Close'].values

# Save data in case no internet
sp_data.to_csv('sp.csv')
import pandas as pd
sp_data = pd.read_csv('sp.csv', index_col=0)
sp_data.index = pd.to_datetime(sp_data.index)
sp_settle = sp_data['Close'].values

# Plot
plt.figure(figsize=(10,4))
plt.plot(sp_data.index, sp_settle, label='Daily Settle')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('S&P Settle')
plt.savefig('sp.png')

# Lag
n_fast = 50
n_slow = 100

# Moving averages
sma_fast = sma(sp_settle, n_fast)
sma_slow = sma(sp_settle, n_slow)

# Plot
plt.figure(figsize=(10,4))
plt.plot(sp_data.index, sp_settle, label='Daily Settle')
plt.plot(sp_data.index, sma_fast, label='%s-day MA'%n_fast)
plt.plot(sp_data.index, sma_slow, label='%s-day MA'%n_slow)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('S&P Settle and Simple Moving Averages')
plt.savefig('SMA.png')

###############################################################################
# EMA trend following
###############################################################################

# Half-life
hl_fast = 12.5
hl_slow = 25

# Moving averages
ema_fast = ema(sp_settle, hl_fast)
ema_slow = ema(sp_settle, hl_slow)

# Plot
plt.figure(figsize=(10,4))
plt.plot(sp_data.index, sp_settle, label='Daily Settle')
plt.plot(sp_data.index, ema_fast, label='Half-life: %s'%hl_fast)
plt.plot(sp_data.index, ema_slow, label='Half-life: %s'%hl_slow)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('S&P Settle and Exponential Moving Averages')
plt.savefig('EMA.png')

###############################################################################
# Signal and taction
###############################################################################

def trend_signal(price, hl_fast, hl_slow):
    """
    Calculate cross over moving average signal.

    Parameters
    ----------
    price : Numpy array
        Input price time series.
    hl_fast : int or float
        Fast filter half-life.
    hl_slow : int or float
        Slow filter half-life.

    Returns
    -------
    signal : Numpy array
        Buy or sell signal.

    """
    ema_fast = ema(price, hl_fast)
    ema_slow = ema(price, hl_slow)

    signal = np.sign(ema_fast - ema_slow)
    
    return signal

def market_returns(price):
    """Caculate market log returns."""
    mktret = price*np.nan
    mktret[1:] = np.log(price[1:]/price[:-1])
    
    return mktret

def calc_traction(signal, mktret):
    """[Traction at (t)] = [signal at (t-1)] * [return at (t)]"""
    
    traction = signal*np.nan
    traction[1:] = signal[:-1]*mktret[1:]
    
    return traction
    
sp_signal = trend_signal(sp_settle, 20, 100)
sp_ret = market_returns(sp_settle)
sp_traction = calc_traction(sp_signal, sp_ret)
start = 100 

# Plot
plt.figure(figsize=(10,4))
plt.plot(sp_data.index[start:], np.nancumsum(sp_traction[start:]))
plt.xlabel('Date')
plt.ylabel('Traction')
plt.title('Cumulative Traction for Trend Signal')
plt.savefig('traction.png')

###############################################################################
# Positioning
###############################################################################
def calc_pos(signal, mktret):
    """Caculate position from signal and market volatility."""
    vol = pd.Series(mktret).rolling(20).std().values
    pos = signal/vol
    
    return pos

def calc_gearing(pnl, vol_target, window):
    """
    Caculate gearing from a volatiliy target

    Parameters
    ----------
    pnl : Numpy array
        PnL time series.
    vol_target : float
        Target volatility.
    window: float
        Rolling volatility window.

    Returns
    -------
    gearing : Numpy array
        Gearing time series.

    """
    gearing = 1/pd.Series(pnl).rolling(window).std().values*vol_target
    
    return gearing

vol_target = np.nanstd(sp_ret)
sp_pos = calc_pos(sp_signal, sp_ret)
sp_pnl = calc_traction(sp_pos, sp_ret)
gearing = calc_gearing(sp_pnl, vol_target, 50)
sp_pnl = sp_pnl*gearing
sp_pos = sp_pos*gearing

# Plot
plt.figure(figsize=(10,4))
plt.plot(sp_data.index[start:], np.nancumsum(sp_pnl[start:]), label='Trend')
plt.plot(sp_data.index[start:], np.nancumsum(sp_ret[start:]), label='Buy & hold')
plt.xlabel('Date')
plt.ylabel('Cumulative RoR')
plt.title('Trend Following S&P PnL, %.2f %% Daily Vol'%(vol_target*100))
plt.legend()
plt.savefig('sp_pnl.png')

###############################################################################
# Bitcoin
###############################################################################

# Download bitcoin data from Yahoo
bc_data = yf.download('BTC-USD', start='2019-02-01', end='2024-02-09')
bc_settle = bc_data['Close'].values

# Save data in case no internet
bc_data.to_csv('bc.csv')
import pandas as pd
bc_data = pd.read_csv('bc.csv', index_col=0)
bc_data.index = pd.to_datetime(bc_data.index)
bc_settle = bc_data['Close'].values

bc_signal = trend_signal(bc_settle, 10, 20)
bc_ret = market_returns(bc_settle)
bc_pos = calc_pos(bc_signal, bc_ret)
bc_pnl = calc_traction(bc_signal, bc_ret)
vol_target = np.nanstd(bc_ret)
gearing = calc_gearing(bc_pnl, vol_target, 50) 
bc_pnl = bc_pnl*gearing

# Plot
plt.figure(figsize=(10,4))
plt.plot(bc_data.index[start:], np.nancumsum(bc_pnl[start:]), label='Trend')
plt.plot(bc_data.index[start:], np.nancumsum(bc_ret[start:]), label='Buy & hold')
plt.xlabel('Date')
plt.ylabel('Cumulative RoR')
plt.title('Trend Following Bitcoin PnL, %.2f %% Daily Vol'%(vol_target*100))
plt.legend()
plt.savefig('bc_pnl.png')

###############################################################################
# Portfolio Construction
###############################################################################
df = sp_data.join(bc_data, lsuffix='_sp', rsuffix='_bc', on='Date', how='inner')

vol_target = 0.01
sp_weight = 0.5
bc_weight = 0.5

sp_settle = df['Close_sp'].values
sp_signal = trend_signal(sp_settle, 20, 100)
sp_ret = market_returns(sp_settle)
sp_pos = calc_pos(sp_signal, sp_ret)
sp_pnl = calc_traction(sp_pos, sp_ret)
sp_gearing = calc_gearing(sp_pnl, vol_target, 50)
sp_pos = sp_pos*sp_gearing*sp_weight
sp_pnl = sp_pnl*sp_gearing*sp_weight

bc_settle = df['Close_bc'].values
bc_signal = trend_signal(bc_settle, 10, 20)
bc_ret = market_returns(bc_settle)
bc_pos = calc_pos(bc_signal, bc_ret)
bc_pnl = calc_traction(bc_pos, bc_ret)
bc_gearing = calc_gearing(bc_pnl, vol_target, 50)
bc_pos = bc_pos*bc_gearing*bc_weight
bc_pnl = bc_pnl*bc_gearing*bc_weight

pos = np.stack([sp_pos, bc_pos], axis=1)
mktret = np.stack([sp_ret, bc_ret], axis=1)
pnl = np.sum(calc_traction(pos, mktret), axis=1)
gearing = calc_gearing(pnl, vol_target, 100)
pnl = pnl*gearing

plt.figure(figsize=(10,4))
plt.plot(df.index[start:], np.nancumsum(pnl[start:]), label='Portfolio')
plt.plot(df.index[start:], np.nancumsum(sp_pnl[start:]), label='S&P')
plt.plot(df.index[start:], np.nancumsum(bc_pnl[start:]), label='Bitcoin')
plt.xlabel('Date')
plt.ylabel('Cumulative RoR')
plt.title('Trend Following Portfolio, %.1f %% Daily Vol'%(vol_target*100))
plt.legend()
plt.savefig('port_pnl.png')
