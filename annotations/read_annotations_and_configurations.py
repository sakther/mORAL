import pandas as pd
import matplotlib.pylab as plt
import math
import statistics
import numpy as np
import os
import datetime
import time
from pytz import timezone
from annotations.CONSTANTS import *


def get_event_configurations():
    D = pd.read_csv(metadata_input_file, index_col=False)

    M_brush_with = dict()
    M_floss_with = dict()
    M_ori_left = dict()
    M_ori_right = dict()
    M_wrist = dict()
    M_is_new_device = dict()

    for i in range(len(D['video_file_name'])):
        key = D['video_file_name'].iloc[i]

        is_normal_brush = int(D['is_manual_brushing'].iloc[i])
        is_oralb_brush = int(D['is_smart_brushing'].iloc[i])
        is_flossing = int(D['is_flossing'].iloc[i])
        left_ori = int(D['left_orientation'].iloc[i])
        right_ori = int(D['right_orientation'].iloc[i])
        is_new_device = int(D['is_new_device'].iloc[i])
        wrist = D['wrist'].iloc[i]
        if wrist in ['L']:
            wrist = LEFT_WRIST
        if wrist in ['R']:
            wrist = RIGHT_WRIST
        if wrist in ['B']:
            wrist = BOTH_WRIST

        if is_normal_brush == 1:
            M_brush_with[key] = MANUAL_BRUSHING
        elif is_oralb_brush == 1:
            M_brush_with[key] = SMART_BRUSHING
        else:
            M_brush_with[key] = 'N/A'

        if is_flossing == 1:
            M_floss_with[key] = D['flossing_with'].iloc[i]
        else:
            M_floss_with[key] = 'N/A'
        M_ori_left[key] = left_ori
        M_ori_right[key] = right_ori
        M_wrist[key] = wrist
        M_is_new_device[key] = is_new_device
    return M_brush_with, M_floss_with, M_ori_left, M_ori_right, M_wrist, M_is_new_device


def get_label_annotations(annotation_filename):
    if os.path.exists(annotation_dir + annotation_filename):
        D = pd.read_csv(annotation_dir + annotation_filename, index_col=False)
        # print(D.columns)
        return D
    else:
        return None


# ---------------------------------------------------------------

def convert_pid_sid_to_eventkeyprefix(pid, sid: str):
    '''
    :param pid: ab3s
    :param sid: d2017-06-18
    :return: ab3s_20170618
    '''
    event_key_prefix = pid + '_' + sid[1:].replace('-', '')
    return event_key_prefix


# print(convert_pid_sid_to_eventkeyprefix('ab3s', 'd2017-06-18'))


def get_orientations_and_device_info(pid, sid):
    '''
    :param pid: 'ab3s'
    :param sid: d2017-06-18
    :return:
    '''
    M_brush_with, M_floss_with, M_ori_left, M_ori_right, M_wrist, M_is_new_device = get_event_configurations()
    event_key_prefix = convert_pid_sid_to_eventkeyprefix(pid, sid)
    event_keys = [v for v in M_ori_left.keys() if v.startswith(event_key_prefix) and M_brush_with[v] == MANUAL_BRUSHING]
    if len(event_keys) == 0:
        return 0, 0, 0
    return M_ori_left[event_keys[0]], M_ori_right[event_keys[0]], M_is_new_device[event_keys[0]]


# ------------------------------------------------------------------------------
def get_overlap_portion(st1, et1, st2, et2):
    overlap_portion = min(et1, et2) - max(st1, st2)
    return overlap_portion


def is_any_overlap_portion(st, et, list2):
    for bStart, bEnd, w, e_key in list2:
        if get_overlap_portion(st, et, bStart, bEnd) > 0:
            return True
    return False


def get_manual_brushing_labels(X: list, mbrushing_segmentList: list, sBrushing_segmentList=[], wrist=RIGHT_WRIST):
    '''
    :param X:
    :param mbrushing_segmentList: list of segment, segment := [starttime, endtime, wrist, eventkey]
    :return:

    Ambiguous labels are assigned as -1.

    '''
    Y = []
    for xi in X:
        st = xi[3]
        et = xi[4]
        yi = 0
        event_key = ''
        if is_any_overlap_portion(st, et, sBrushing_segmentList) > 0: #if smart brushing are selected as candidate
            yi = -1
        else:
            for bStart, bEnd, w, e_key in mbrushing_segmentList:
                if get_overlap_portion(st, et, bStart, bEnd) > 0:
                    if wrist == RIGHT_WRIST and w in [RIGHT_WRIST, BOTH_WRIST]:
                        yi = 1
                        event_key = e_key
                        break
                    elif wrist == LEFT_WRIST and w == LEFT_WRIST:
                        yi = 1
                        event_key = e_key
                        break
                    else:
                        yi = -1
                        break
                elif get_overlap_portion(st, et, bStart - 180, bEnd + 180) > 0:
                    yi = -1
                    break

        Y.append(yi)
        # # if yi == 1:
        # #     print('found brushing', yi)
        # Y.append([yi, event_key])
    return Y


def get_flossing_labels(X: list, stringFlossing_segmentList: list):
    '''
    :param X:
    :param stringFlossing_segmentList: list of segment, segment := [starttime, endtime, wrist]
    :return:
    '''
    Y = []
    for xi in X:
        st = xi[2]
        et = xi[3]
        yi = 0
        for bStart, bEnd in stringFlossing_segmentList:
            if get_overlap_portion(st, et, bStart, bEnd):
                yi = 1
                break
        Y.append(yi)
    return Y


# def get_annotations_eventwise(annotation_filename):
#     event_key = annotation_filename[:-5]
#     M_brush_with, M_floss_with, M_ori_left, M_ori_right, M_wrist, M_is_new_device = get_event_configurations()
#
#     D = get_label_annotations(annotation_filename)
#
#     mbrushing_start_end_times, stringFlossing_start_end_times = [], []
#
#     for i in range(len(D['label'])):
#         st = D['start_timestamp'].iloc[i]
#         et = D['end_timestamp'].iloc[i]
#         lb = D['label'].iloc[i]
#         if lb == BRUSHING:
#             if M_brush_with[event_key] == MANUAL_BRUSHING:
#                 mbrushing_start_end_times.append([st, et])
#         if lb == FLOSSING:
#             if M_floss_with[event_key] == STRING_FLOSSING:
#                 stringFlossing_start_end_times.append([st, et])
#     return mbrushing_start_end_times, stringFlossing_start_end_times


def get_brushing_flossing_segments(pid):
    mBrushing_segmentList, sBrushing_segmentList, stringFlossing_segmentList = [], [], []
    M_brush_with, M_floss_with, M_ori_left, M_ori_right, M_wrist, M_is_new_device = get_event_configurations()

    annotation_files = [d for d in os.listdir(annotation_dir) if d.startswith(pid)]
    for annotation_file in annotation_files:

        if '_ext' in annotation_file:
            continue
        event_key = annotation_file[:-4]

        D = get_label_annotations(annotation_file)
        for i in range(len(D['label'])):
            st = D['start_timestamp'].iloc[i] / 1000.0
            et = D['end_timestamp'].iloc[i] / 1000.0
            lb = D['label'].iloc[i]
            if lb == MANUAL_BRUSHING:
                mBrushing_segmentList.append([st, et, M_wrist[event_key], event_key])
            if lb == SMART_BRUSHING:
                sBrushing_segmentList.append([st, et, M_wrist[event_key], event_key])
            if lb == STRING_FLOSSING:
                stringFlossing_segmentList.append([st, et, M_wrist[event_key], event_key])

    return mBrushing_segmentList, sBrushing_segmentList, stringFlossing_segmentList


def get_manual_brushing_annotation_eventkeys(pid):
    mBrushing_eventkeys = []
    annotation_files = [d for d in os.listdir(annotation_dir) if d.startswith(pid)]
    for annotation_file in annotation_files:

        if '_ext' in annotation_file:
            continue
        event_key = annotation_file[:-4]

        D = get_label_annotations(annotation_file)
        for i in range(len(D['label'])):
            if D['label'].iloc[i] == MANUAL_BRUSHING:
                mBrushing_eventkeys.append(event_key)
                break
    mBrushing_eventkeys = list(set(mBrushing_eventkeys))

    return mBrushing_eventkeys

# print(get_label_annotations('9a6b_20170420_000900.csv'))
