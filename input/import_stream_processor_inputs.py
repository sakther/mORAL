from typing import List
import os
from utils.merge_two_datastream import *
from utils.hand_orientation import *
from utils.orientation_mapping import *
from annotations.CONSTANTS import *
# LEFT_WRIST = 'leftwrist'
# RIGHT_WRIST = 'rightwrist'

# ax_left_filename = 'left-wrist-accelx.csv'
# ay_left_filename = 'left-wrist-accely.csv'
# az_left_filename = 'left-wrist-accelz.csv'
# gx_left_filename = 'left-wrist-gyrox.csv'
# gy_left_filename = 'left-wrist-gyroy.csv'
# gz_left_filename = 'left-wrist-gyroz.csv'
# ax_right_filename = 'right-wrist-accelx.csv'
# ay_right_filename = 'right-wrist-accely.csv'
# az_right_filename = 'right-wrist-accelz.csv'
# gx_right_filename = 'right-wrist-gyrox.csv'
# gy_right_filename = 'right-wrist-gyroy.csv'
# gz_right_filename = 'right-wrist-gyroz.csv'
#
# import pytz
# tz = pytz.timezone('US/Central')


def convert_sample(sample):
    return list([float(x.strip()) for x in sample.split(',')])


def line_parser(input):
    ts, sample = input.split(',')
    start_time = int(float(ts)) / 1000.0

    sample = float(sample)

    return [start_time, sample]


def line_parser_offset(input):
    ts, offset, sample = input.split(',', 2)
    start_time = int(float(ts)) / 1000.0

    sample = convert_sample(sample)
    if len(sample) == 1:
        sample = sample[0]

    return sample


def load_data_offset(filename):
    fp = open(filename)
    file_content = fp.read()
    fp.close()

    lines = file_content.splitlines()
    data = list(map(line_parser_offset, lines))

    return data


def load_data(filename):
    if not os.path.exists(filename):
        return []

    fp = open(filename)
    file_content = fp.read()
    fp.close()

    lines = file_content.splitlines()
    data = list(map(line_parser, lines[1:]))

    return data


def get_accelerometer(data_dir, wrist, ori, is_new_device=False):

    accel = []
    if wrist in [LEFT_WRIST]:
        if os.path.isfile(data_dir + ax_left_filename):
            ax = load_data(os.path.join(data_dir, ax_left_filename))
            # ax = load_data(data_dir + ax_left_filename)
            ay = load_data(os.path.join(data_dir, ay_left_filename))
            az = load_data(os.path.join(data_dir, az_left_filename))
            if is_new_device:
                fac = left_ori_newdevice[ori]
                accel = [[v[0], fac[0]*ay[i][1], fac[1]*v[1], fac[2]*az[i][1]] for i, v in enumerate(ax)]
            else:
                fac = left_ori[ori]
                accel = [[v[0], fac[0]*v[1], fac[1]*ay[i][1], fac[2]*az[i][1]] for i, v in enumerate(ax)]

    else:
        fac = right_ori[ori]
        if os.path.isfile(data_dir + ax_right_filename):
            ax = load_data(os.path.join(data_dir, ax_right_filename))
            ay = load_data(os.path.join(data_dir, ay_right_filename))
            # ay = [[v[0], -v[1]] for v in ay]
            az = load_data(os.path.join(data_dir, az_right_filename))
            if is_new_device:
                fac = right_ori_newdevice[ori]
                accel = [[v[0], fac[0]*ay[i][1], fac[1]*v[1], fac[2]*az[i][1]] for i, v in enumerate(ax)]
            else:
                fac = right_ori[ori]
                accel = [[v[0], fac[0]*v[1], fac[1]*ay[i][1], fac[2]*az[i][1]] for i, v in enumerate(ax)]
    return accel


def get_gyroscope(data_dir, wrist, ori, is_new_device=False):
    if wrist in [LEFT_WRIST]:
        gx = load_data(os.path.join(data_dir, gx_left_filename))
        gy = load_data(os.path.join(data_dir, gy_left_filename))
        gz = load_data(os.path.join(data_dir, gz_left_filename))
        if is_new_device:
            fac = left_ori_newdevice[ori]
            gyro = [[v[0], fac[0]*gy[i][1], fac[1]*v[1], fac[2]*gz[i][1]] for i, v in enumerate(gx)]
        else:
            fac = left_ori[ori]
            gyro = [[v[0], fac[0]*v[1], fac[1]*gy[i][1], fac[2]*gz[i][1]] for i, v in enumerate(gx)]
    else:
        gx = load_data(os.path.join(data_dir, gx_right_filename))
        gy = load_data(os.path.join(data_dir, gy_right_filename))
        # gy = [[v[0], -v[1]] for v in gy]

        gz = load_data(os.path.join(data_dir, gz_right_filename))
        if is_new_device:
            fac = right_ori_newdevice[ori]
            gyro = [[v[0], fac[0]*gy[i][1], fac[1]*v[1], fac[2]*gz[i][1]] for i, v in enumerate(gx)]
        else:
            fac = right_ori[ori]
            gyro = [[v[0], fac[0]*v[1], fac[1]*gy[i][1], fac[2]*gz[i][1]] for i, v in enumerate(gx)]
    return gyro

def get_magnitude(ax, ay, az):
    return math.sqrt(ax * ax + ay * ay + az * az)


def get_accel_gyro_mag_orientation(data_dir, wrist, ori, is_new_device=False):
    '''
    :param data_dir:
    :param wrist:
    :return:list of (t, ax, ay, az, gx, gy, gz, Amag, Gmag, roll, pitch, yaw, Omag)
    '''
    accel = get_accelerometer(data_dir, wrist, ori, is_new_device)
    gyro = get_gyroscope(data_dir, wrist, ori, is_new_device)
    if len(accel) == 0 or len(gyro) == 0:
        return []
    AG = merge_accel_gyro(accel, gyro)
    Amag = [get_magnitude(v[1], v[2], v[3]) for v in AG]
    Gmag = [get_magnitude(v[4], v[5], v[6]) for v in AG]
    AGM = list(np.column_stack((AG, Amag, Gmag)))

    AGMO = apply_complementary_filter(AGM)
    Omag = [get_magnitude(v[9], v[10], v[11]) for v in AGMO]
    AGMO = list(np.column_stack((AGMO, Omag)))
    return AGMO


# ---- FOR GROUNDTRUTH -----
import pickle

# def get_ground_truth(pid):
#     video_dur_dir = 'C:/Users/nsleheen/DATA/ROBAS_MEMPHIS/Video_dur/'
#     print(pid)
#     D = pickle.load( open( video_dur_dir + pid[:3] + 'vd_dur.pkl', "rb" ) )
#     return D

def get_overlap_portion(st1, et1, st2, et2):
    overlap_portion = min(et1, et2) - max(st1, st2)
    return overlap_portion


def get_brushing_label_of_interval(st, et, D):
    '''

    :param st:
    :param et:
    :param D:
    :return: labels: 0: not brushing, -1: not brushing but within video, 1: 75% brushing-- no pause, 10: 90% brushing with pause,
    '''
    event_keys = D.keys()
    yi = 0
    for event_key in event_keys:
        normal_brushing_times, oralb_brushing_times, string_flossing_times, picks_flossing_times, rinsing_times, pause_times, video_time = D[event_key]
        vst = video_time[0][0]/1000.0
        vet = video_time[0][1]/1000.0
        if get_overlap_portion(st, et, vst, vet) > 0:
            yi = -1
        if len(normal_brushing_times) == 0:
            continue
        bst = min([v[0]/1000.0 for v in normal_brushing_times])
        bet = max([v[1]/1000.0 for v in normal_brushing_times])
        # print(event_key, bst, bet, st, et)
        paused_dur = 0
        for pt in pause_times:
            paused_in_brushing = get_overlap_portion(bst, bet, pt[0]/1000.0, pt[1]/1000.0)
            paused_in_window = get_overlap_portion(st, et, pt[0]/1000.0, pt[1]/1000.0)
            if paused_in_brushing > 0 and paused_in_window > 0:
                paused_dur += paused_in_window

        overlap_portion = get_overlap_portion(st, et, bst, bet)
        if overlap_portion>0 and overlap_portion - paused_dur > (et-st)*0.75:
            yi =1
            break
        elif overlap_portion>0 and overlap_portion > (et-st)*0.90:
            yi = 10
            break
    return yi


def check_lable(normal_brushing_times, pause_times, st, et):
    yi=0
    if len(normal_brushing_times) == 0:
        return yi
    nbst = min([v[0]/1000.0 for v in normal_brushing_times])
    nbet = max([v[1]/1000.0 for v in normal_brushing_times])
    paused_dur = 0
    for pt in pause_times:
        paused_in_nbrushing = get_overlap_portion(nbst, nbet, pt[0]/1000.0, pt[1]/1000.0)
        paused_in_window = get_overlap_portion(st, et, pt[0]/1000.0, pt[1]/1000.0)
        if (paused_in_nbrushing > 0) and paused_in_window > 0:
            paused_dur += paused_in_window

    overlap_portion_nb = get_overlap_portion(st, et, nbst, nbet)
    if overlap_portion_nb- paused_dur > (et-st)*0.75:
        yi =1
    elif overlap_portion_nb > (et-st)*0.90:
        yi = 11
    return yi

def get_brushing_flossing_labels_of_interval(st, et, D):
    '''

    :param st:
    :param et:
    :param D:
    :return: labels: 0: not brushing, -1: not brushing but within video
    , 1: 75% brushing-- no pause, 10: 90% brushing with pause,
    '''
    event_keys = D.keys()
    yi = 0
    for event_key in event_keys:
        normal_brushing_times, oralb_brushing_times, string_flossing_times, picks_flossing_times, rinsing_times, pause_times, video_time = D[event_key]
        vst = 5*60- video_time[0][0]/1000.0
        vet = 10*60+video_time[0][1]/1000.0
        if get_overlap_portion(st, et, vst, vet) <= 0:
            continue
        else:
            yi=-1

            y_tmp = check_lable(normal_brushing_times, pause_times, st, et)
            if y_tmp > 0:
                if y_tmp == 1:
                    yi = 1
                else:
                    yi = 11

            y_tmp = check_lable(oralb_brushing_times, pause_times, st, et)
            if y_tmp > 0:
                if y_tmp == 1:
                    yi = 2
                else:
                    yi = 12

            y_tmp = check_lable(string_flossing_times, pause_times, st, et)
            if y_tmp > 0:
                if y_tmp == 1:
                    yi = 3
                else:
                    yi = 13

            y_tmp = check_lable(picks_flossing_times, pause_times, st, et)
            if y_tmp > 0:
                if y_tmp == 1:
                    yi = 4
                else:
                    yi = 14
            break
    return yi

def get_oral_labels(X: list, D: dict):
    '''

    :param X:
    :param D: map of all event_keys to annotation
    :return:
    '''
    Y = []
    X_new = []
    for xi in X:
        st = xi[2]
        et = xi[3]
        yi = get_brushing_flossing_labels_of_interval(st, et, D)
        Y.append(yi)
    return Y

def get_normal_brushing_labels(X: list, D: dict):
    '''

    :param X:
    :param D: map of all event_keys to annotation
    :return:
    '''
    Y = []
    X_new = []
    for xi in X:
        st = xi[2]
        et = xi[3]
        yi = get_brushing_label_of_interval(st, et, D)
        Y.append(yi)
    return Y

def get_flossing_labels(X, D):
    Y = []
    for xi in X:
        st = xi[2]
        et = xi[3]
        yi = 0
        D = D[D['label']== 'flossing']
        #         print(D)
        stimes = list(D['start_timestamp'])
        etimes = list(D['end_timestamp'])

        for i, value in enumerate(stimes):
            bst = stimes[i]/1000.0
            bet = etimes[i]/1000.0
            #             print("st et",st,et )
            #             print("bst, bet", bst, bet)
            overlap_portion = min(bet, et) - max(st, bst)

            if overlap_portion>0 and overlap_portion > (et-st)*0.75:
                yi =1
            # if st >= bst and et <= bet:
            #     yi =1
        Y.append(yi)

    return Y

# ---------------RININS________________________
def get_rinsing_label_of_interval(st, et, D):
    '''

    :param st:
    :param et:
    :param D:
    :return: labels: 0: not brushing, -1: not brushing but within video, 1: 75% brushing-- no pause, 10: 90% brushing with pause,
    '''
    event_keys = D.keys()
    yi = 0
    for event_key in event_keys:
        normal_brushing_times, oralb_brushing_times, string_flossing_times, picks_flossing_times, rinsing_times, pause_times, video_time = D[event_key]
        vst = video_time[0][0]/1000.0
        vet = video_time[0][1]/1000.0
        if get_overlap_portion(st, et, vst, vet) > 0:
            yi = -1
            for rt in rinsing_times:
                rst =rt[0]/1000.0
                ret =rt[1]/1000.0

                overlap_portion = get_overlap_portion(st, et, rst, ret)
                if overlap_portion>0:
                    yi =1
            return yi
    return yi

def get_rinsing_labels(X: list, D: dict):
    '''
    :param X:
    :param D: map of all event_keys to annotation
    :return:
    '''
    Y = []
    X_new = []
    for xi in X:
        st = xi[2]
        et = xi[3]
        yi = get_rinsing_label_of_interval(st, et, D)
        Y.append(yi)
    return Y