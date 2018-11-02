"""
    TrendsPy project
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    :copyright: © 2018, AppliedAlpha.com
    :author: Dmitry E. Marienko
    :license: GPL
"""

from typing import Union

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from .indicators import bollinger, bollinger_atr


def binarize(_x, n, limits=(None, None), center=False):
    n0 = n // 2 if center else 0
    _min = np.min(_x) if limits[0] is None else limits[0]
    _max = np.max(_x) if limits[1] is None else limits[1]
    return np.floor(n * (_x - _min) / (_max - _min)) - n0


def slope_ols(x):
    x = x[~np.isnan(x)]
    xs = 2 * (x - min(x)) / (max(x) - min(x)) - 1
    m = OLS(xs, np.vander(np.linspace(-1, 1, len(xs)), 2)).fit()
    return m.params[0]


def slope_angle(p, t):
    return 180 * np.arctan(p / t) / np.pi


def scaling_transform(x, n=5, need_round=True, limits=None):
    if limits is None:
        _lmax = max(abs(x))
        _lmin = -_lmax
    else:
        _lmax = max(limits)
        _lmin = min(limits)

    if need_round:
        ni = np.round(np.interp(x, (_lmin, _lmax), (-2 * n, +2 * n))) / 2
    else:
        ni = np.interp(x, (_lmin, _lmax), (-n, +n))
    return pd.Series(ni, index=x.index)


def rolling_series_slope(x: pd.Series, period: Union[str, int], n_bins=5, method='ols', scaling='transform'):
    """
    Detects timeseries slope on rolling basis.
    There are 2 methods supported for getting slope of timseries: ordinary least squares (ols) and as arctan of price angle
    OLS method usually shows more smoothed measure of slope compared to angle.
    """
    if method == 'ols':
        slp_meth = lambda z: slope_ols(z)
        _lmts = (-1, 1)
    elif method == 'angle':
        slp_meth = lambda z: slope_angle(z[-1] - z[0], len(z))
        _lmts = (-90, 90)
    else:
        raise ValueError('Unknown Method %s' % method)

    _min_p = period
    if isinstance(period, str):
        _min_p = 1  # pd.Timedelta(period).days

    roll_slope = x.rolling(period, min_periods=_min_p).apply(slp_meth)
    if scaling == 'transform':
        return scaling_transform(roll_slope, n=n_bins, limits=_lmts)
    elif scaling == 'binarize':
        return binarize(roll_slope, n=(n_bins - 1) * 4, limits=_lmts, center=True) / 2

    return roll_slope


def trend_detector(data, period, nstd, avg='sma', k_ext=1, exit_on_mid=False,
                   use_atr=False, atr_period=12, atr_avg='kama') -> pd.DataFrame:
    """
    Trend detector method

    :param data: input series/frame
    :param period: bb period
    :param nstd: bb num of stds
    :param avg: averaging ma type
    :param k_ext: extending factor
    :param exit_on_mid: trend is over when x crosses middle of bb
    :param use_atr: true if we use bollinger_atr for trend detecting
    :param atr_period: ATR period (used only when use_atr is True)
    :param atr_avg: ATR smoother (used only when use_atr is True)
    :return: frame
    """
    # flatten list lambda
    flatten = lambda l: [item for sublist in l for item in sublist]

    # just taking close prices
    x = data.close if isinstance(data, pd.DataFrame) else data

    if use_atr:
        midle, smax, smin = bollinger_atr(data, period, atr_period, nstd, avg, atr_avg)
    else:
        midle, smax, smin = bollinger(x, period, nstd, avg)

    trend = (((x > smax.shift(1)) + 0.0) - ((x < smin.shift(1)) + 0.0)).replace(0, np.nan)

    # some special case if we want to exit when close is on the opposite side of median price
    if exit_on_mid:
        lom, him = ((x < midle).values, (x > midle).values)
        t = 0;
        _t = trend.values.tolist()
        for i in range(len(trend)):
            t0 = _t[i]
            t = t0 if np.abs(t0) == 1 else t
            if (t > 0 and lom[i]) or (t < 0 and him[i]):
                t = 0
            _t[i] = t
        trend = pd.Series(_t, trend.index)
    else:
        trend = trend.fillna(method='ffill').fillna(0.0)

    # making resulting frame
    m = x.to_frame().copy()
    m['trend'] = trend
    m['blk'] = (m.trend.shift(1) != m.trend).astype(int).cumsum()
    m['x'] = abs(m.trend) * (smax * (-m.trend + 1) - smin * (1 + m.trend)) / 2
    _g0 = m.reset_index().groupby(['blk', 'trend'])
    m['x'] = flatten(abs(_g0['x'].apply(np.array).transform(np.minimum.accumulate).values))
    m['utl'] = m.x.where(m.trend > 0)
    m['dtl'] = m.x.where(m.trend < 0)

    # signals
    tsi = pd.DatetimeIndex(_g0['time'].apply(lambda x: x.values[0]).values)
    m['uts'] = m.loc[tsi].utl
    m['dts'] = m.loc[tsi].dtl

    return m.filter(items=['uts', 'dts', 'trend', 'utl', 'dtl'])
