"""
    TrendsPy project
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: © 2018, AppliedAlpha.com
    :author: Dmitry E. Marienko
    :license: GPL
"""

def p_dist(a, b):
    """
    Distances between two patterns elements
    :param a: element 1
    :param b: element 2
    :return:
    """
    P = {
     '0': (0,0), '': (0,0), '-': (0,0),
     'H': (1, 1), 'DH': (2, 2), 'UH': (3, 3), 'LH': (4, 4), 'HH':(5, 5),
     'L': (1,-1), 'UL': (2,-2), 'DL': (3,-3), 'HL': (4,-4), 'LL':(5,-5),
    }
    xa, ya = P[a]
    xb, yb = P[b]
    return (xa - xb)**2 + (ya - yb)**2


def dist(x, y):
    if len(x) != len(y):
        raise ValueError('Vectors must be equal sizes !')
    return sum([p_dist(i, j) for i, j in zip(x, y)])


def _phases(z):
    """
    Split symbolic pattern presentation into 2 phases: initial (I), evolution (E) and finish (F)
    :param z: symbolic patterns presentation (for example 'LL, U, L, HH, UL, H')
    :return: phases
    """
    zs = [x.strip().upper() for x in z.split(',')]
    _ex = sorted([zs.index('LL'), zs.index('HH')])
    Ex = slice(_ex[0], _ex[1] + 1)
    Ix = slice(0, Ex.start)
    Fx = slice(Ex.stop, len(zs))
    return zs[Ix], zs[Ex], zs[Fx]


def _align_left_or_right(a, b, right):
    if len(a) == 0: a = ['0']
    if len(b) == 0: b = ['0']
    n_a, n_b = len(a), len(b)
    if n_a == n_b: return a, b

    if n_a > n_b:
        aa = a
        bb = ['0'] * n_a
        if right:
            bb[-1:-(n_b + 1):-1] = b[::-1]
        else:
            bb[:n_b] = b
    else:
        aa = ['0'] * n_b
        bb = b
        if right:
            aa[-1:-(n_a + 1):-1] = a[::-1]
        else:
            aa[:n_a] = a
    return aa, bb


def _align_center(a, b):
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2 or n_a % 2 > 0 or n_b % 2 > 0:
        raise ValueError('Wrong number input elements, must be >= 2 and even. Check: %s / %s' % (str(a), str(b)))

    if not (((a[0].startswith('LL') or a[0].startswith('HH')) and (a[-1].startswith('LL') or a[-1].startswith('HH')))
            or ((b[0].startswith('LL') or b[0].startswith('HH')) and (
                    b[-1].startswith('LL') or b[-1].startswith('HH')))):
        raise ValueError('Check start and end elements !')

    if n_a == n_b: return a, b

    if n_a > n_b:
        aa, bb = a, ['0'] * n_a
        bb[0], bb[-1] = b[0], b[-1]
        if n_b > 2:
            bb[1:n_b // 2] = b[1:n_b // 2]
            bb[-n_b // 2:-1] = b[-n_b // 2:-1]
    else:
        aa, bb = ['0'] * n_b, b
        aa[0], aa[-1] = a[0], a[-1]
        if n_a > 2:
            aa[1:n_a // 2] = a[1:n_a // 2]
            aa[-n_a // 2:-1] = a[-n_a // 2:-1]
    return aa, bb


def pattern_dist(x, y, wI=1, wE=1, wF=1):
    """
    Measures distance between two patterns in symbolic representation

    Example:

    >>> pattern_dist('H, L, H, LL, HH, UL', 'LL, H, L, H, L, HH, UL')
    
    14

    :param x: symbolis pattern 1
    :param y: symbolis pattern 2
    :param wI: weight for initial phase
    :param wE: weight for evolution phase
    :param wF: weight for finish phase
    :return: distance
    """
    xI, xE, xF = _phases(x)
    yI, yE, yF = _phases(y)

    return wI * dist(*_align_left_or_right(xI, yI, True)) + \
           wE * dist(*_align_center(xE, yE)) + \
           wF * dist(*_align_left_or_right(xF, yF, False))
