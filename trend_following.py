# -*- coding: utf-8 -*-
"""
Trend following example.

@author: Katie He
"""
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

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
sp_data = yf.download('^GSPC', start='2019-02-01', end='2025-01-24')
sp_data.columns = sp_data.columns.droplevel(level=1)
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
# Exponential weights
###############################################################################
hl = np.arange(10, 110, 10)
l = np.arange(101)

# Plot
plt.figure(figsize=(10,5))
for h in hl:
    alpha = 1-(1/2)**(1/h)
    w = alpha*(1-alpha)**(l)
    plt.plot(l, w, label='Half-life: %d, sum of weights:%.2f'%(h, np.sum(w)))
plt.xlabel('Lag')
plt.ylabel('Weight')
plt.legend()
plt.title('Exponential Weights with Different Half-lives')
plt.savefig('alpha_weights.png')

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
    
sp_signal = trend_signal(sp_settle, 10, 20)
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

def sharpe_ratio(pnl, risk_free=0):
    """
    Calculate sharpe ratio

    Parameters
    ----------
    pnl : Numpy array
        PnL time series.
    risk_free : float, optional
        Daily risk free rate. The default is 0.

    Returns
    -------
    s : float
        Sharpe ratio of time series.

    """
    
    s = np.nanmean(pnl-risk_free)/np.nanstd(pnl)*(256**0.5)
    return s

vol_target = np.nanstd(sp_ret)
sp_pos = calc_pos(sp_signal, sp_ret)
sp_pnl = calc_traction(sp_pos, sp_ret)
gearing = calc_gearing(sp_pnl, vol_target, 50)
sp_pnl = sp_pnl*gearing
sp_pos = sp_pos*gearing

window = 50
traction_vol = pd.Series(sp_traction).rolling(window).std().values
pnl_vol = pd.Series(sp_pnl).rolling(window).std().values

# Plot
plt.figure(figsize=(10,4))
plt.plot(sp_data.index[start:], np.nancumsum(sp_pnl[start:]),
         label='Trend, sharpe=%.2f'%sharpe_ratio(sp_pnl[start:]))
plt.plot(sp_data.index[start:], np.nancumsum(sp_ret[start:]),
         label='Buy & hold, sharpe=%.2f'%sharpe_ratio(sp_ret[start:]))
plt.xlabel('Date')
plt.ylabel('Cumulative RoR')
plt.title('Trend Following S&P PnL, %.2f %% Daily Vol'%(vol_target*100))
plt.legend()
plt.savefig('sp_pnl.png')

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
ax[0].plot(sp_data.index[start:], np.nancumsum(sp_traction[start:]),
         label='Traction, sharpe=%.2f'%sharpe_ratio(sp_traction[start:]))
ax[0].plot(sp_data.index[start:], np.nancumsum(sp_pnl[start:]),
         label='Geared PnL, sharpe=%.2f'%sharpe_ratio(sp_pnl[start:]))
ax[0].legend()
ax[0].set_ylabel('Cumulative RoR')
ax[1].plot(sp_data.index[start:], traction_vol[start:], label='Traction')
ax[1].plot(sp_data.index[start:], pnl_vol[start:], label='Geared PnL')
ax[1].set_xlabel('Date')
ax[1].set_ylabel('Daily vol')
ax[1].legend()
fig.suptitle('Effect of Gearing')
fig.savefig('gearing.png')

###############################################################################
# Bitcoin
###############################################################################

# Download bitcoin data from Yahoo
bc_data = yf.download('BTC-USD', start='2019-02-01', end='2025-01-24')
bc_data.columns = bc_data.columns.droplevel(level=1)
bc_settle = bc_data['Close'].values

bc_signal = trend_signal(bc_settle, 1, 10)
bc_ret = market_returns(bc_settle)
bc_pos = calc_pos(bc_signal, bc_ret)
bc_pnl = calc_traction(bc_signal, bc_ret)
vol_target = np.nanstd(bc_ret)
gearing = calc_gearing(bc_pnl, vol_target, 50) 
bc_pnl = bc_pnl*gearing

# Plot
plt.figure(figsize=(10,4))
plt.plot(bc_data.index[start:], np.nancumsum(bc_pnl[start:]),
         label='Trend, sharpe=%.2f'%sharpe_ratio(bc_pnl[start:]))
plt.plot(bc_data.index[start:], np.nancumsum(bc_ret[start:]),
         label='Buy & hold, sharpe=%.2f'%sharpe_ratio(bc_ret[start:]))
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
sp_signal = trend_signal(sp_settle, 10, 20)
sp_ret = market_returns(sp_settle)
sp_pos = calc_pos(sp_signal, sp_ret)
sp_pnl = calc_traction(sp_pos, sp_ret)
sp_gearing = calc_gearing(sp_pnl, vol_target, 50)
sp_pos = sp_pos*sp_gearing*sp_weight
sp_pnl = sp_pnl*sp_gearing*sp_weight

bc_settle = df['Close_bc'].values
bc_signal = trend_signal(bc_settle, 1, 10)
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
pos = pos*gearing.reshape(-1,1)
pnl = pnl*gearing

plt.figure(figsize=(10,4))
plt.plot(df.index[start:], np.nancumsum(pnl[start:]),
         label='Portfolio, sharpe=%.2f'%sharpe_ratio(pnl[start:]))
plt.plot(df.index[start:], np.nancumsum(sp_pnl[start:]),
         label='S&P, sharpe=%.2f'%sharpe_ratio(sp_pnl[start:]))
plt.plot(df.index[start:], np.nancumsum(bc_pnl[start:]),
         label='Bitcoin, sharpe=%.2f'%sharpe_ratio(bc_pnl[start:]))
plt.xlabel('Date')
plt.ylabel('Cumulative RoR')
plt.title('Trend Following Portfolio, %.1f %% Daily Vol'%(vol_target*100))
plt.legend()
plt.savefig('port_pnl.png')
