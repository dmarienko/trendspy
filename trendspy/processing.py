import pandas as pd
import numpy as np
from datetime import datetime


def pullbacks_estimate(h5, n_ema_days=14, f=0.1):
    h1d = ohlc_resample(h5, '1D')

    # averaged daily spread (on 14 days rolling window)
    return f * apply_to_frame(ema, 100 * (h1d.high - h1d.low) / h1d.low, n_ema_days).fillna(method='bfill')


def run_detection_algo(h5, pb_method: str, pb_window=14, pb_factor=0.1, pb_fixed=np.inf,
                       sess_start='18:05:00', sess_end='16:30:00', sess_cutoff='9:40:00'):
    h1d = ohlc_resample(h5, '1D')
    days = sorted(set(h1d.index.date))

    pb = None
    if pb_method.startswith('roll'):
        pb = pullbacks_estimate(h5, pb_window, pb_factor)
    elif pb_method.startswith('fixed'):
        if not np.isinf(pb_fixed) and abs(pb_fixed) <= 1.0:
            pb = pd.Series(100.0 * abs(pb_fixed), index=days)
        else:
            raise ValueError('For fixed method you must pass numeric positive value (<= 1.0) to pb_fixed argument')

    # convert to deltas
    sess_start_delta = pd.to_timedelta(sess_start)
    sess_end_delta = pd.to_timedelta(sess_end)
    sess_cutoff_time = (lambda x: datetime.time(x.hours, x.minutes, x.seconds))(pd.to_timedelta(sess_cutoff).components)

    # here we need to shift back to previos day
    if sess_start_delta > sess_end_delta:
        days_shift = pd.Timedelta('1D')
    else:
        days_shift = pd.Timedelta(0)

    pats = {}
    px = pd.concat((h1d, pb.shift(1).rename('th')), axis=1)

    _p_i = 0
    for d in px.index[1:]:
        if d.weekday() != 6:  # we skip Sundays
            t0 = d - days_shift

            # from sess_start to sess_end
            hh = h5[pd.to_datetime(t0) + sess_start_delta: pd.to_datetime(d) + sess_end_delta]
            _, h_log = detect_price_patterns_log(hh.close, pulback_pct=px.loc[d].th / 100)

            # select first one before 9:40
            if h_log is not None:
                _dt = h_log.index[h_log.index >= pd.to_datetime(d)]
                _pr = _dt[_dt.time < sess_cutoff_time]
                if len(_pr) > 0:
                    last_p = _pr[-1]
                    p = ','.join(h_log.loc[last_p].dropna().values)
                    pats[last_p] = p

        if (_p_i % 100) == 0: print('.', end='')
        _p_i += 1

    print(' [OK]')
    return pd.Series(pats)