from typing import Union

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS


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
