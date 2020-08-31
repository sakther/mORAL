import numpy as np
import math
from scipy.stats import skew
from scipy.stats import kurtosis

def zero_cross_rate(series):
    """
    How often the signal changes sign (+/-)
    """
    series_mean = np.mean(series)
    series = [v-series_mean for v in series]
    zero_cross_count = (np.diff(np.sign(series)) != 0).sum()
    # print('zero_cross_count', zero_cross_count)
    return zero_cross_count / len(series)

def compute_statistical_features(data):
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    skewness = skew(data)
    kurt = kurtosis(data)
    power = np.mean([v * v for v in data])
    zc = zero_cross_rate(data)
    return [mean, median, std, skewness, kurt, power, zc]
