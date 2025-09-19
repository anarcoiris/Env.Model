import numpy as np
from scipy import stats

def psi(expected, actual, buckets=10):
    # Population Stability Index
    e_counts, bins = np.histogram(expected, bins=buckets)
    a_counts, _ = np.histogram(actual, bins=bins)
    e_percents = e_counts / max(len(expected), 1)
    a_percents = a_counts / max(len(actual), 1)
    e_percents = np.where(e_percents == 0, 1e-6, e_percents)
    a_percents = np.where(a_percents == 0, 1e-6, a_percents)
    return np.sum((e_percents - a_percents) * np.log(e_percents / a_percents))

def ks_test(expected, actual):
    try:
        stat, p = stats.ks_2samp(expected, actual)
        return stat, p
    except Exception:
        return None, None