"""
    TrendsPy project
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: © 2018, AppliedAlpha.com
    :author: Dmitry E. Marienko
    :license: GPL
"""

import pandas as pd
import numpy as np

from .utils import column_vector, sink_nans_down, lift_nans_up, rolling_sum, shift, nans


def sma(x, period):
    """
    Classical simple moving average

    :param x: input data (as np.array or pd.DataFrame/Series)
    :param period: period of smoothing
    :return: smoothed values
    """
    if period <= 0:
        raise ValueError('Period must be positive and greater than zero !!!')

    x = column_vector(x)
    x, ix = sink_nans_down(x, copy=True)
    s = rolling_sum(x, period) / period
    return lift_nans_up(s, ix)


def ema(x, span, init_mean=True, min_periods=0) -> np.ndarray:
    """
    Exponential moving average

    :param x: data to be smoothed
    :param span: number of data points for smooth
    :param init_mean: use average of first span points as starting ema value (default is true)
    :param min_periods: minimum number of observations in window required to have a value (0)
    :return:
    """
    x = column_vector(x)
    alpha = 2.0 / (1 + span)

    # for every column move all starting nans to the end and here we copy inputs to avoid modifying input series
    x, ix = sink_nans_down(x, copy=True)

    s = np.zeros(x.shape)
    start_i = 1
    if init_mean:
        s += np.nan
        s[span - 1, :] = np.mean(x[:span, :], axis=0)
        start_i = span
    else:
        s[0, :] = x[0, :]

    for i in range(start_i, x.shape[0]):
        s[i, :] = alpha * x[i, :] + (1 - alpha) * s[i - 1, :]

    if min_periods > 0:
        s[:min_periods - 1, :] = np.nan

    # lift up 'starting' nans
    return lift_nans_up(s, ix)


def zlema(x: np.ndarray, n: int, init_mean=True):
    """
    'Zero lag' moving average
    :type x: np.array
    :param x:
    :param n:
    :param init_mean: True if initial ema value is average of first n points
    :return:
    """
    return ema(2 * x - shift(x, n), n, init_mean=init_mean)


def dema(x, n: int, init_mean=True):
    """
    Double EMA

    :param x:
    :param n:
    :param init_mean: True if initial ema value is average of first n points
    :return:
    """
    e1 = ema(x, n, init_mean=init_mean)
    return 2 * e1 - ema(e1, n, init_mean=init_mean)


def tema(x, n: int, init_mean=True):
    """
    Triple EMA

    :param x:
    :param n:
    :param init_mean: True if initial ema value is average of first n points
    :return:
    """
    e1 = ema(x, n, init_mean=init_mean)
    e2 = ema(e1, n, init_mean=init_mean)
    return 3 * e1 - 3 * e2 + ema(e2, n, init_mean=init_mean)


def kama(x, period, fast_span=2, slow_span=30):
    """
    Kaufman Adaptive Moving Average

    :param x: input data (as np.array or pd.DataFrame/Series)
    :param period: period of smoothing
    :param fast_span: fast period (default is 2 as in canonical impl)
    :param slow_span: slow period (default is 30 as in canonical impl)
    :return: smoothed values
    """
    x, ix = sink_nans_down(column_vector(x), copy=True) # here we need copy incoming values
    if period >= len(x) - max(ix):
        raise ValueError('Wrong value for period. period parameter must be less than number of input observations')
    abs_diff = np.abs(x - shift(x, 1))
    er = np.abs(x - shift(x, period)) / rolling_sum(np.nan_to_num(abs_diff), period)
    sc = np.square((er * (2.0 / (fast_span + 1) - 2.0 / (slow_span + 1.0)) + 2 / (slow_span + 1.0)))
    ama = nans(sc.shape)

    # here ama_0 = x_0
    ama[period - 1, :] = x[period - 1, :]
    for i in range(period, len(ama)):
        ama[i, :] = ama[i - 1, :] + sc[i, :] * (x[i, :] - ama[i - 1, :])

    # drop 1-st kama value (just for compatibility with ta-lib)
    ama[period - 1, :] = nans(x.shape[1])

    # we do not copy outputs but modify it for decreasing latency time
    return lift_nans_up(ama, ix)


def atr(x, window=14, smoother='sma'):
    """
    Average True Range indicator

    :param x: input series
    :param window: smoothing window size
    :param smoother: smooting method: sma, ema, zlema, tema, dema, kama
    :return:
    """
    if not (isinstance(x, pd.DataFrame) and sum(x.columns.isin(['open', 'high', 'low', 'close'])) == 4):
        raise ValueError("Input series must be DataFrame within 'open', 'high', 'low' and 'close' columns defined !")

    smoothers = {'ema': ema, 'tema': tema, 'dema': dema, 'zlema': zlema, 'kama': kama}

    h_l = abs(x['high'] - x['low'])
    h_pc = abs(x['high'] - x['close'].shift(1))
    l_pc = abs(x['low'] - x['close'].shift(1))
    tr = pd.concat((h_l, h_pc, l_pc), axis=1).max(axis=1)

    # not smoothed
    tr_s = tr

    if smoother == 'sma':
        tr_s = tr.rolling(window=window).mean()
    elif smoother in smoothers:
        tr_s = pd.Series(smoothers[smoother](tr, window).flatten(), x.index)

    return tr_s


def rolling_std_with_mean(x, mean, window):
    """
    Calculates rolling standard deviation for data from x and already calculated mean series
    :param x: series data
    :param mean: calculated mean
    :param window: window
    :return: rolling standard deviation
    """
    return np.sqrt((((x - mean) ** 2).rolling(window=window).sum() / (window - 1)))


def bollinger(x, window=14, nstd=2, mean='sma'):
    """
    Bollinger Bands indicator

    :param x: input data
    :param window: lookback window
    :param nstd: number of standard devialtions for bands
    :param mean: method for calculating mean: sma, ema, tema, dema, zlema, kama
    :return: mean, upper and lower bands
    """
    methods = {'ema': ema, 'tema': tema, 'dema': dema, 'zlema': zlema, 'kama': kama}

    if mean == 'sma':
        rolling_mean = x.rolling(window=window).mean()
        rolling_std = x.rolling(window=window).std()
    elif mean in methods:
        rolling_mean = pd.Series(methods[mean](x, window).flatten(), x.index)
        rolling_std = rolling_std_with_mean(x, rolling_mean, window)
    else:
        raise ValueError("Method '%s' is not supported" % mean)

    upper_band = rolling_mean + (rolling_std * nstd)
    lower_band = rolling_mean - (rolling_std * nstd)

    return rolling_mean, upper_band, lower_band


def bollinger_atr(x, window=14, atr_window=14, natr=2, mean='sma', atr_mean='ema'):
    """
    Bollinger Bands indicator where ATR is used for bands range estimating
    :param x: input data
    :param window: window size for averaged price
    :param atr_window: atr window size
    :param natr:  number of ATRs for bands
    :param mean: method for calculating mean: sma, ema, tema, dema, zlema, kama
    :param atr_mean:  method for calculating mean for atr: sma, ema, tema, dema, zlema, kama
    :return: mean, upper and lower bands
    """
    if not (isinstance(x, pd.DataFrame) and sum(x.columns.isin(['open', 'high', 'low', 'close'])) == 4):
        raise ValueError("Input series must be DataFrame within 'open', 'high', 'low' and 'close' columns defined !")

    b, _, _ = bollinger(x.close, window, 0, mean)
    a = natr * atr(x, atr_window, atr_mean)

    return b, b + a, b - a
