"""
    TrendsPy project
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: © 2018, AppliedAlpha.com
    :author: Dmitry E. Marienko
    :license: GPL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

from trendspy.charting.mpl_finance import ohlc_plot


def plot_trends(trends, uc='w--', dc='c--', lw=0.7):
    """
    Plot local trends from frame containing trends data
    :param trends:
    """
    if not trends.empty:
        u, d = trends.UpTrends.dropna(), trends.DownTrends.dropna()
        plt.plot([u.index, u.end], [u.start_price, u.end_price], uc, lw=lw, marker='.', markersize=5);
        plt.plot([d.index, d.end], [d.start_price, d.end_price], dc, lw=lw, marker='.', markersize=5);


def plot_markup(x, trends, logs, number_of_log_record):
    number_of_log_record = min(len(logs) - 1, number_of_log_record)
    rec = logs.iloc[number_of_log_record]
    tp = rec.name
    labels = rec.dropna().values
    ts = trends[:tp]

    # plot series data
    if isinstance(x, pd.DataFrame) and all([z in x.columns for z in ['open', 'close', 'high', 'low']]):
        ohlc_plot(x[trends.index[0]: max(trends.index[-1], logs.index[-1])])
    else:
        plt.plot(x[trends.index[0]: max(trends.index[-1], logs.index[-1])], lw=2)

    # plot trends
    plot_trends(ts[:-1])

    plt.plot([tp, tp], [plt.ylim()[0], plt.ylim()[1]], 'r--')
    j = 0
    for r in ts.iterrows():
        ti = r[0]
        px = r[1]['UpTrends'].start_price
        is_up = isinstance(px, (float, int))
        px = px if is_up else r[1]['DownTrends'].start_price
        c = 'y' if is_up else 'w'
        plt.annotate(labels[j], xy=(ti, px), xytext=(0, -10 if is_up else 10), textcoords='offset points',
                     horizontalalignment='right', verticalalignment='top', fontsize=7, color=c)
        j += 1
        if j >= len(labels): break
    plt.title('%s: %s' % (str(tp), ','.join(labels)), fontsize=9)


def mark_reversals(sd, small_m=np.nan):
    """
    Go through all reversals points from sd and find highest high, lowest lowm, lowest high, highest low,
    up high/low, down high/low labels.
    low -> L, high -> H,
    lowest low -> LL,
    highest high -> HH,
    lowest high -> LH,
    highest low -> HL,
    up high or low -> UH or UL
    down high or low -> DH or DL
    
    :param sd: dictionary contains reversal points. Key is time, value - tuple (s, y, magnitude)
               s - symbol ('H' - local high, 'L' - local low),
               y - series value of local extremum
               magnitude - percentage change from previous extremum (y[this]/y[prev] - 1)
    :param small_m: if it's finite it attaches additional measure to each pattern:
           X_s if magnitude < small_m, X_b if magnitude > small_m
    :return: 
    """
    # prepare frame
    z = pd.DataFrame.from_dict(sd, orient='index')
    z.columns = ['t', 'p', 'm']

    # higher high and lower low
    ihh, ill = z[z.t == 'H'].p.idxmax(), z[z.t == 'L'].p.idxmin()
    z.loc[ihh] = ('HH', z.loc[ihh].p, z.loc[ihh].m)
    z.loc[ill] = ('LL', z.loc[ill].p, z.loc[ill].m)

    m_classes = None
    if len(sd) > 2:
        # lower high and higher low
        _s0 = z[z.t == 'H']
        if not _s0.empty:
            ilh = _s0.p.idxmin()
            z.loc[ilh] = ('Hl', z.loc[ilh].p, z.loc[ilh].m)

        _s0 = z[z.t == 'L']
        if not _s0.empty:
            ihl = _s0.p.idxmax()
            z.loc[ihl] = ('Lh', z.loc[ihl].p, z.loc[ihl].m)

        u, m_classes = {}, {}
        _l, _lp, _h, _hp = None, 0, None, 0
        for r in z.iterrows():
            t, p, m = r[1]
            is_not_extr = not t in ['LL', 'HH', 'Hl', 'Lh']

            # append magnitude class (small, big)
            m_classes[r[0]] = 's' if (abs(m) <= small_m and m != 0) else 'b' if abs(m) > small_m else '-'

            if t[0] == 'L':
                if is_not_extr and _l and p > _lp:
                    u[r[0]] = ('LU', p, m)
                elif is_not_extr and _l and p < _lp:
                    u[r[0]] = ('LD', p, m)
                _l, _lp = t, p

            if t[0] == 'H':
                if is_not_extr and _h and p > _hp:
                    u[r[0]] = ('HU', p, m)
                elif is_not_extr and _h and p < _hp:
                    u[r[0]] = ('HD', p, m)
                _h, _hp = t, p

        if len(u) > 0:
            zu = pd.DataFrame.from_dict(u, orient='index')
            zu.columns = z.columns
            z.loc[zu.index] = zu

    z = z.replace({'Hl': 'LH', 'Lh': 'HL', 'HU': 'UH', 'HD': 'DH', 'LU': 'UL', 'LD': 'DL'})
    return pd.concat((z, pd.Series(m_classes, name='mclass')), axis=1).fillna('-')


def detect_price_patterns_log(x, pulback_pct=0.75, use_prev_movement_size_for_percentage=False, classify=False):
    """

    :param x: series
    :param pulback_pct: pullback's percentage (0...1)
    :param use_prev_movement_size_for_percentage: how to calculate percentage of pullback
    :param classify: if True it will add classe of prev. motion magnitude (_s - small, _b - big)
    :return: (trends data, dataframe with logged patterns)
    """
    # check input arguments
    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    # drop nans (not sure about 0 as replacement)
    if x.hasnans:
        x = x.fillna(0)

    extremums = OrderedDict()
    log_rec = OrderedDict()
    class_small_th = 2 * pulback_pct

    mi, mx, direction = 0, 0, 0
    i_drops, i_grows = [], []
    timeline = x.index

    for i in range(1, len(x)):
        v = x.iat[i]

        if direction <= 0:
            if v < x.iat[mi]:
                mi = i
                direction = -1
            else:
                # floating up
                if use_prev_movement_size_for_percentage:
                    l_mv = pulback_pct * (x.iat[mx] - x.iat[mi])
                else:
                    l_mv = pulback_pct * x.iat[mi]

                # check condition
                if l_mv < v - x.iat[mi]:
                    # new low occured
                    i_drops.append([mx, mi])
                    ti = timeline[i]

                    # when we got first low we also need to memoize first high
                    if len(extremums) == 0:
                        extremums[timeline[mx]] = ('H', x.iat[mx], 0)

                    extremums[ti] = ('L', x.iat[mi], x.iat[mi] / x.iat[mx] - 1)
                    f = mark_reversals(extremums, class_small_th)
                    if classify:
                        log_rec[ti] = [('%s_%s' % (k, v) if v != '-' else k) for k, v in
                                       zip(f.t.values, f.mclass.values)]
                    else:
                        log_rec[ti] = f.t.values

                    # new high pretender and reverse direction
                    mx = i
                    direction = 1

        if direction >= 0:
            if v > x.iat[mx]:
                mx = i
                direction = +1
            else:
                if use_prev_movement_size_for_percentage:
                    l_mv = pulback_pct * (x.iat[mx] - x.iat[mi])
                else:
                    l_mv = pulback_pct * x.iat[mx]

                if l_mv < x.iat[mx] - v:
                    # new high occured
                    i_grows.append([mi, mx])
                    ti = timeline[i]

                    # when we got first high we also need to memoize first low
                    if len(extremums) == 0:
                        extremums[timeline[mi]] = ('L', x.iat[mi], 0)

                    # store found high
                    extremums[ti] = ('H', x.iat[mx], x.iat[mx] / x.iat[mi] - 1)
                    f = mark_reversals(extremums, class_small_th)
                    if classify:
                        log_rec[ti] = [('%s_%s' % (k, v) if v != '-' else k) for k, v in
                                       zip(f.t.values, f.mclass.values)]
                    else:
                        log_rec[ti] = f.t.values

                    # new low pretender and reverse direction
                    mi = i
                    direction = -1

    i_drops, i_grows = np.array(i_drops), np.array(i_grows)

    # Nothing is found
    if len(i_drops) == 0 or len(i_grows) == 0:
        print("\n\t[WARNING] find_movements: No trends found for given conditions !")
        return pd.DataFrame({'UpTrends': [], 'DownTrends': []}), None

    i_d, i_g = x.index[i_drops], x.index[i_grows]
    x_d, x_g = x[i_drops], x[i_grows]

    # drops and grows magnitudes
    v_drops, v_grows = [], []
    if i_drops.size:
        v_drops = abs(x[i_drops[:, 1]].values - x[i_drops[:, 0]].values)
    if i_grows.size:
        v_grows = abs(x[i_grows[:, 1]].values - x[i_grows[:, 0]].values)

    d = pd.DataFrame(OrderedDict({
        'start_price': x_d[:, 0],
        'end_price': x_d[:, 1],
        'delta': v_drops,
        'end': i_d[:, 1]
    }), index=i_d[:, 0])

    g = pd.DataFrame(OrderedDict({
        'start_price': x_g[:, 0],
        'end_price': x_g[:, 1],
        'delta': v_grows,
        'end': i_g[:, 1]
    }), index=i_g[:, 0])

    return pd.concat((g, d), axis=1, keys=['UpTrends', 'DownTrends']), pd.DataFrame.from_dict(log_rec, orient='index')
