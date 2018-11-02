import pandas as pd
import numpy as np

from .utils import column_vector, sink_nans_down, lift_nans_up, rolling_sum, shift


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
