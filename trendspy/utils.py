from ctypes import Union

import numpy as np
import pandas as pd


def column_vector(x):
    """
    Convert any vector to column vector. Matrices remain unchanged.

    :param x: vector
    :return: column vector
    """
    if isinstance(x, (pd.DataFrame, pd.Series)): x = x.values
    return np.reshape(x, (x.shape[0], -1))


def shift(xs: np.ndarray, n: int, fill=np.nan) -> np.ndarray:
    """
    Shift data in numpy array (aka lag function):

    shift(np.array([[1.,2.],
                    [11.,22.],
                    [33.,44.]]), 1)

    >> array([[ nan,  nan],
              [  1.,   2.],
              [ 11.,  22.]])

    :param xs:
    :param n:
    :param fill: value to use for
    :return:
    """
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = fill
        e[n:] = xs[:-n]
    else:
        e[n:] = fill
        e[:n] = xs[-n:]
    return e


def sink_nans_down(x_in, copy=False) -> (np.ndarray, np.ndarray):
    """
    Move all starting nans 'down to the bottom' in every column.

    NaN = np.nan
    x = np.array([[NaN, 1, NaN],
                  [NaN, 2, NaN],
                  [NaN, 3, NaN],
                  [10,  4, NaN],
                  [20,  5, NaN],
                  [30,  6, 100],
                  [40,  7, 200]])

    x1, nx = sink_nans_down(x)
    print(x1)

    >> [[  10.    1.  100.]
        [  20.    2.  200.]
        [  30.    3.   nan]
        [  40.    4.   nan]
        [  nan    5.   nan]
        [  nan    6.   nan]
        [  nan    7.   nan]]

    :param x_in: numpy 1D/2D array
    :param copy: set if need to make copy input to prevent being modified [False by default]
    :return: modified x_in and indexes
    """
    x = np.copy(x_in) if copy else x_in
    n_ix = np.zeros(x.shape[1])
    for i in range(0, x.shape[1]):
        f_n = np.where(~np.isnan(x[:, i]))[0]
        if len(f_n) > 0:
            if f_n[0] != 0:
                x[:, i] = np.concatenate((x[f_n[0]:, i], x[:f_n[0], i]))
            n_ix[i] = f_n[0]
    return x, n_ix


def lift_nans_up(x_in, n_ix, copy=False) -> np.ndarray:
    """
    Move all ending nans 'up to top' of every column.

    NaN = np.nan
    x = np.array([[NaN, 1, NaN],
                  [NaN, 2, NaN],
                  [NaN, 3, NaN],
                  [10,  4, NaN],
                  [20,  5, NaN],
                  [30,  6, 100],
                  [40, 7, 200]])

    x1, nx = sink_nans_down(x)
    print(x1)

    >> [[  10.    1.  100.]
        [  20.    2.  200.]
        [  30.    3.   nan]
        [  40.    4.   nan]
        [  nan    5.   nan]
        [  nan    6.   nan]
        [  nan    7.   nan]]

    x2 = lift_nans_up(x1, nx)
    print(x2)

    >> [[  nan    1.   nan]
        [  nan    2.   nan]
        [  nan    3.   nan]
        [  10.    4.   nan]
        [  20.    5.   nan]
        [  30.    6.  100.]
        [  40.    7.  200.]]

    :param x_in: numpy 1D/2D array
    :param n_ix: indexes for every column
    :param copy: set if need to make copy input to prevent being modified [False by default]
    :return: modified x_in
    """
    x = np.copy(x_in) if copy else x_in
    for i in range(0, x.shape[1]):
        f_n = int(n_ix[i])
        if f_n != 0:
            x[:, i] = np.concatenate((x[-f_n:, i], x[:-f_n, i]))
    return x


def add_constant(x, const=1., prepend=True):
    """
    Adds a column of constants to an array

    Parameters
    ----------
    :param data: column-ordered design matrix
    :param prepend: If true, the constant is in the first column.  Else the constant is appended (last column)
    :param const: constant value to be appended (default is 1.0)
    :return:
    """
    x = column_vector(x)
    if prepend:
        r = (const * np.ones((x.shape[0], 1)), x)
    else:
        r = (x, const * np.ones((x.shape[0], 1)))
    return np.hstack(r)


def isscalar(x):
    """
    Returns true if x is scalar value

    :param x:
    :return:
    """
    return not isinstance(x, (list, tuple, dict, np.ndarray))


def nans(dims):
    """
    nans((M,N,P,...)) is an M-by-N-by-P-by-... array of NaNs.

    :param dims: dimensions tuple
    :return: nans matrix
    """
    return np.nan * np.ones(dims)


def apply_to_frame(func, x, *args, **kwargs):
    """
    Utility applies given function to x and converts result to incoming type

    >>> from ira.analysis.timeseries import ema
    >>> apply_to_frame(ema, data['EURUSD'], 50)
    >>> apply_to_frame(lambda x, p1: x + p1, data['EURUSD'], 1)

    :param func: function to map
    :param x: input data
    :param args: arguments of func
    :param kwargs: named arguments of func
    :return: result of function's application
    """
    if func is None or not isinstance(func, types.FunctionType):
        raise ValueError(str(func) + ' must be callable object')

    xp = column_vector(func(x, *args, **kwargs))
    _name = func.__name__ + '_' + '_'.join([str(i) for i in args])

    if isinstance(x, pd.DataFrame):
        c_names = ['%s_%s' % (c, _name) for c in x.columns]
        return pd.DataFrame(xp, index=x.index, columns=c_names)
    elif isinstance(x, pd.Series):
        return pd.Series(xp.flatten(), index=x.index, name=_name)

    return xp


def ohlc_resample(df, new_freq: str = '1H', vmpt: bool = False) -> Union[pd.DataFrame, dict]:
    """
    Resample OHLCV/tick series to new timeframe.

    Example:
    >>> d = pd.DataFrame({
    >>>          'open' : np.random.randn(30),
    >>>          'high' : np.random.randn(30),
    >>>          'low' : np.random.randn(30),
    >>>          'close' : np.random.randn(30)
    >>>         }, index=pd.date_range('2000-01-01 00:00', freq='5Min', periods=30))
    >>>
    >>> ohlc_resample(d, '15Min')

    :param df: input ohlc or bid/ask quotes or dict
    :param new_freq: how to resample rule (see pandas.DataFrame::resample)
    :param vmpt: use volume weighted price for quotes (if false mid price will be used)
    :return: resampled ohlc / dict
    """

    def __mx_rsmpl(d, freq: str, is_vmpt: bool = False) -> pd.DataFrame:
        _cols = d.columns

        # if we have bid/ask frame
        if 'ask' in _cols and 'bid' in _cols:
            # if sizes are presented we can calc vmpt if need
            if is_vmpt and 'askvol' in _cols and 'bidvol' in _cols:
                mp = (d.ask * d.bidvol + d.bid * d.askvol) / (d.askvol + d.bidvol)
                return mp.resample(freq).agg('ohlc')

            # if there is only asks and bids and we don't need vmpt
            return d[['ask', 'bid']].mean(axis=1).resample(freq).agg('ohlc')

        # for OHLC case or just simple series
        if all([i in _cols for i in ['open', 'high', 'low', 'close']]) or isinstance(d, pd.Series):
            ohlc_rules = {'open': 'first',
                          'high': 'max',
                          'low': 'min',
                          'close': 'last',
                          'ask_vol': 'sum',
                          'bid_vol': 'sum',
                          'volume': 'sum'
                          }
            return d.resample(freq).apply(dict(i for i in ohlc_rules.items() if i[0] in d.columns)).dropna()

        raise ValueError("Can't recognize structure of input data !")

    if isinstance(df, (pd.DataFrame, pd.Series)):
        return __mx_rsmpl(df, new_freq, vmpt)
    elif isinstance(df, dict):
        return {k: __mx_rsmpl(v, new_freq, vmpt) for k, v in df.items()}
    else:
        raise ValueError('Type [%s] is not supported in ohlc_resample' % str(type(df)))


def rolling_sum(x: np.ndarray, n: int) -> np.ndarray:
    """
    Fast running sum for numpy array (matrix) along columns.

    Example:
    >>> rolling_sum(column_vector(np.array([[1,2,3,4,5,6,7,8,9], [11,22,33,44,55,66,77,88,99]]).T), n=5)

    array([[  nan,   nan],
       [  nan,   nan],
       [  nan,   nan],
       [  nan,   nan],
       [  15.,  165.],
       [  20.,  220.],
       [  25.,  275.],
       [  30.,  330.],
       [  35.,  385.]])

    :param x: input data
    :param n: rolling window size
    :return: rolling sum for every column preceded by nans
    """
    ret = np.cumsum(x, axis=0, dtype=float)
    ret[n:, :] = ret[n:, :] - ret[:-n, :]
    return np.concatenate((nans([n - 1, x.shape[1]]), ret[n - 1:, :]))
