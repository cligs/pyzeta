#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: experimental.py
# version: 0.1.0
# source: https://github.com/maslinych/linis-scripts/blob/master/rbo_calc.py
# ported to Python 3 and slightly modified by Albin Zehe

def calc_rbo(l1, l2, p=0.98):
    """ 
    Returns RBO indefinite rank similarity metric, as described in:
    Webber, W., Moffat, A., & Zobel, J. (2010). 
    A similarity measure for indefinite rankings. 
    ACM Transactions on Information Systems.
    doi:10.1145/1852102.1852106.
    """
    sl, ll = sorted([(len(l1), l1), (len(l2), l2)])
    s, S = sl
    l, L = ll

    # Calculate the overlaps at ranks 1 through l 
    # (the longer of the two lists)
    ss = set([])
    ls = set([])
    overs = {}
    for i in range(l):
        ls.add(L[i])
        if i < s:
            ss.add(S[i])
        X_d = len(ss.intersection(ls))
        d = i + 1
        overs[d] = float(X_d)

    # (1) \sum_{d=1}^l (X_d / d) * p^d
    sum1 = 0
    for i in range(l):
        d = i + 1
        sum1 += overs[d] / d * pow(p, d)
    X_s = overs[s]
    X_l = overs[l]

    # (2) \sum_{d=s+1}^l [(X_s (d - s)) / (sd)] * p^d
    sum2 = 0
    for i in range(s, l):
        d = i + 1
        sum2 += (X_s * (d - s) / (s * d)) * pow(p, d)

    # (3) [(X_l - X_s) / l + X_s / s] * p^l
    sum3 = ((X_l - X_s) / l + X_s / s) * pow(p, l)

    # Equation 32. 
    rbo_ext = (1 - p) / p * (sum1 + sum2) + sum3
    return rbo_ext


if __name__ == "__main__":
    list1 = ['A', 'B', 'C', 'D', 'E', 'H']
    list2 = ['D', 'B', 'F', 'A']
    print
    calc_rbo(list1, list2, 0.98)
