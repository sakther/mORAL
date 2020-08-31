import numpy as np
import math
from scipy.stats import skew
from scipy.stats import kurtosis

from features.fft_features import *
from features.filter_based_on_time_duration import filter_based_on_duration
from features.time_domain_features import *
from annotations.CONSTANTS import *

def create_featurenames_for_brushing():
    featurenames = ['pid', 'date', 'wrist(L/R/B)', 'stime', 'etime', 'duration']
    streanames = ['ax', 'ay', 'az', 'max_accl', 'gx', 'gy', 'gz', 'roll', 'pitch', 'yaw']
    features = ['mean', 'median', 'std', 'skewness', 'kurt', 'power', 'zero_crossing', 'fft_centroid', 'fft_spread',
                'spec_entropy', 'spec_entropy_old', 'fft_Flux', 'spec_rolloff']
    # features = ['mean', 'median', 'std', 'skewness', 'kurt', 'power', 'zero_crossing', 'fft_min', 'fft_max', 'fft_mean', 'fft_std', 'fft_kurtosis', 'spec_entropy']

    for sn in streanames:
        for fn in features:
            featurenames.append(sn + ':' + fn)
    corr_features = ['c_ax_ay', 'c_ax_az', 'c_ay_az', 'c_gx_gy', 'c_gx_gz', 'c_gy_gz', 'mse_ax_ay', 'mse_ax_az',
                     'mse_ay_az', 'mse_gx_gy', 'mse_gx_gz', 'mse_gy_gz']
    featurenames.extend(corr_features)

    # print('#of features', len(featurenames))

    return featurenames


def create_featurenames_for_flossing():
    featurenames = ['pid', 'date', 'wrist(L/R/B)', 'stime', 'etime', 'duration']
    streanames = ['ax', 'ay', 'az', 'max_accl', 'gx', 'gy', 'gz', 'roll', 'pitch', 'yaw']
    features = ['mean', 'median', 'std', 'skewness', 'kurt', 'power', 'zero_crossing', 'fft_centroid', 'fft_spread',
                'spec_entropy', 'spec_entropy_old', 'fft_Flux', 'spec_rolloff']
    # features = ['mean', 'median', 'std', 'skewness', 'kurt', 'power', 'zero_crossing', 'fft_min', 'fft_max', 'fft_mean', 'fft_std', 'fft_kurtosis', 'spec_entropy']

    for sn in streanames:
        for fn in features:
            featurenames.append(sn + ':' + fn)
    corr_features = ['c_ax_ay', 'c_ax_az', 'c_ay_az', 'c_gx_gy', 'c_gx_gz', 'c_gy_gz', 'mse_ax_ay', 'mse_ax_az',
                     'mse_ay_az', 'mse_gx_gy', 'mse_gx_gz', 'mse_gy_gz']

    corr_crosswrist_features = ['c_roll', 'c_pitch', 'c_yaw', 'c_a_mag', 'c_g_mag', 'mse_roll', 'mse_pitch', 'mse_yaw',
                                'mse_a_mag', 'mse_g_mag']

    featurenames.extend(corr_features)
    featurenames.extend(corr_crosswrist_features)

    print('#of features', len(featurenames))

    return featurenames


def get_magnitude(ax, ay, az):
    return math.sqrt(ax * ax + ay * ay + az * az)


def get_event_correlation(A, B):
    corr = np.corrcoef(A, B)
    return corr[0, 1]


def standardize(df):
    """
    Make the mean of data 0 and normalize
    """
    return (df - df.mean()) / df.std()


def compute_power(data):
    power = np.mean([v * v for v in data])
    return power


def get_MSE(A, B):
    sqDiff = [(t - p) ** 2 for t, p in zip(A, B)]
    return sum(sqDiff) / len(A)


# (t, ax, ay, az, gx, gy, gz, Amag, Gmag, roll, pitch, yaw)
def compute_correlation_accel_gyro_features(AGMO_r):
    data_r = np.array(AGMO_r)

    c_ax_ay = get_event_correlation(data_r[:, 1], data_r[:, 2])
    c_ax_az = get_event_correlation(data_r[:, 1], data_r[:, 3])
    c_ay_az = get_event_correlation(data_r[:, 2], data_r[:, 2])

    c_gx_gy = get_event_correlation(data_r[:, 4], data_r[:, 5])
    c_gx_gz = get_event_correlation(data_r[:, 4], data_r[:, 6])
    c_gy_gz = get_event_correlation(data_r[:, 5], data_r[:, 6])

    mse_ax_ay = get_MSE(data_r[:, 1], data_r[:, 2])
    mse_ax_az = get_MSE(data_r[:, 1], data_r[:, 3])
    mse_ay_az = get_MSE(data_r[:, 2], data_r[:, 2])

    mse_gx_gy = get_MSE(data_r[:, 4], data_r[:, 5])
    mse_gx_gz = get_MSE(data_r[:, 4], data_r[:, 6])
    mse_gy_gz = get_MSE(data_r[:, 5], data_r[:, 6])

    return [c_ax_ay, c_ax_az, c_ay_az, c_gx_gy, c_gx_gz, c_gy_gz, mse_ax_ay, mse_ax_az, mse_ay_az, mse_gx_gy, mse_gx_gz,
            mse_gy_gz]


# (t, ax, ay, az, gx, gy, gz, Amag, Gmag, roll, pitch, yaw)
def compute_correlation_bothwrist_features(AGMO_l, AGMO_r):
    data_l = np.array(AGMO_l)
    data_r = np.array(AGMO_r)

    c_roll = get_event_correlation(data_l[:, 9], data_r[:, 9])
    c_pitch = get_event_correlation(data_l[:, 10], data_r[:, 10])
    c_yaw = get_event_correlation(data_l[:, 11], data_r[:, 11])
    c_a_mag = get_event_correlation(data_l[:, 7], data_r[:, 7])
    c_g_mag = get_event_correlation(data_l[:, 8], data_r[:, 8])

    mse_roll = get_MSE(data_l[:, 9], data_r[:, 9])
    mse_pitch = get_MSE(data_l[:, 10], data_r[:, 10])
    mse_yaw = get_MSE(data_l[:, 11], data_r[:, 11])
    mse_a_mag = get_MSE(data_l[:, 7], data_r[:, 7])
    mse_g_mag = get_MSE(data_l[:, 8], data_r[:, 8])

    return [c_roll, c_pitch, c_yaw, c_a_mag, c_g_mag, mse_roll, mse_pitch, mse_yaw, mse_a_mag, mse_g_mag]


def get_all_features_of_one_window_BRUSHING(data):
    f = compute_statistical_features(data)
    f_tmp = fouriar_features(data)
    f.extend(f_tmp)
    return f


# (t, ax, ay, az, gx, gy, gz, Amag, Gmag, roll, pitch, yaw)
def get_window_features(AGMO):
    data = np.array(AGMO)
    f_ax = get_all_features_of_one_window_BRUSHING(data[:, 1])
    f_ay = get_all_features_of_one_window_BRUSHING(data[:, 2])
    f_az = get_all_features_of_one_window_BRUSHING(data[:, 3])
    f_max_accl = [max([x, y, z]) for x, y, z in zip(f_ax, f_ay, f_az)]

    f_gx = get_all_features_of_one_window_BRUSHING(data[:, 4])
    f_gy = get_all_features_of_one_window_BRUSHING(data[:, 5])
    f_gz = get_all_features_of_one_window_BRUSHING(data[:, 6])
    f_roll = get_all_features_of_one_window_BRUSHING(data[:, 9])
    f_ptch = get_all_features_of_one_window_BRUSHING(data[:, 10])
    f_yaw = get_all_features_of_one_window_BRUSHING(data[:, 11])
    f_corr = compute_correlation_accel_gyro_features(data)
    f = []
    f.extend(f_ax)
    f.extend(f_ay)
    f.extend(f_az)
    f.extend(f_max_accl)
    f.extend(f_gx)
    f.extend(f_gy)
    f.extend(f_gz)
    f.extend(f_roll)
    f.extend(f_ptch)
    f.extend(f_yaw)
    f.extend(f_corr)
    # f = list(np.column_stack((f_ax, f_ay, f_az, f_max_accl, f_gx, f_gy, f_gz, f_roll, f_ptch, f_yaw, f_corr)))

    return f


def generate_all_window_and_compute_brushing_features(pid, sid, AGMO, cands, wrist=RIGHT_WRIST) -> object:
    '''
    :param pid:
    :param sid:
    :param AGMO: tuple of (t, ax, ay, az, gx, gy, gz, Amag, Gmag, roll, pitch, yaw)
    :param cands:
    :return:
    '''

    all_features = []
    # cands = filter_based_on_duration(cands, AGMO)

    for cand in cands:
        start_index = cand[0]
        end_index = cand[1]
        stime = AGMO[start_index][0]
        etime = AGMO[end_index][0]

        AGMO_r_win = AGMO[start_index:end_index]
        feature_vector = get_window_features(AGMO_r_win)

        f = [pid, sid, wrist, stime, etime, etime - stime]
        f.extend(feature_vector)
        all_features.append(f)

    return all_features


def generate_all_window_and_compute_flossing_features(pid, sid, AGMO_l, AGMO_r, cands, wrist='R') -> object:
    '''
    :param pid:
    :param sid:
    :param AGMO_l: tuple of (t, ax, ay, az, gx, gy, gz, Amag, Gmag, roll, pitch, yaw)
    :param cands:
    :return:
    '''

    all_features = []
    cur_index = 0

    for cand in cands:
        start_index = cand[0]
        end_index = cand[1]
        stime = AGMO_l[start_index][0]
        etime = AGMO_l[end_index][0]

        AGMO_r_win = AGMO_r[start_index:end_index]
        AGMO_l_win = AGMO_l[start_index:end_index]
        feature_vector = get_window_features(AGMO_r_win)

        feature_vector.extend(compute_correlation_bothwrist_features(AGMO_l_win, AGMO_r_win))

        f = [pid, sid, wrist, stime, etime, etime - stime]
        f.extend(feature_vector)
        all_features.append(f)

    return all_features
