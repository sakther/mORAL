import pandas as pd
import matplotlib.pylab as plt
import math
import statistics
import numpy as np
import os
import datetime
import time
from pytz import timezone

NORMAL_BRUSHING = 'normal_brush'
ORALB_BRUSHING = 'oralB_brush'
STRING_FLOSSING = 'string'

metadata_filename = '\\\\MD2K_LAB\\md2k_lab_share\\Data\\ROBAS-Memphis\\from_phone_storage\\META_DATA_video_new.xlsx'
metadata_filename = 'Y:\\Data\\ROBAS\\from_phone_storage\\META_DATA_video_new.xlsx'
col_name = ['video_name', 'pid', 'wrist', 'is_normal_brush', 'is_oralb_brush', 'is_flossing', 'flossing_with',
            'is_rinsing' , 'duration', 'is_video_paused',	'left_orientation',	'right_orientation']

def get_flossing_with(video_event_key):
    M_brush_with, M_floss_with, M_rins_with, M_ori_left, M_ori_right = get_event_METADATA()
    return M_floss_with[video_event_key]

def get_brushing_with(video_event_key):
    M_brush_with, M_floss_with, M_rins_with, M_ori_left, M_ori_right = get_event_METADATA()
    return M_brush_with[video_event_key]

def get_rinsing_with(video_event_key):
    M_brush_with, M_floss_with, M_rins_with, M_ori_left, M_ori_right = get_event_METADATA()
    return M_rins_with[video_event_key]

def get_left_orientation(video_event_key):
    M_brush_with, M_floss_with, M_rins_with, M_ori_left, M_ori_right = get_event_METADATA()
    return int(M_ori_left[video_event_key])

def get_right_orientation(video_event_key):
    M_brush_with, M_floss_with, M_rins_with, M_ori_left, M_ori_right = get_event_METADATA()
    return int(M_ori_right[video_event_key])

def convert_pid_sid_to_eventkeyprefix(pid, sid: str):
    '''
    :param pid: 001
    :param sid: s2017-06-18
    :return: 001_20170618
    '''
    event_key_prefix = pid + '_' + sid[1:].replace('-', '')
    return event_key_prefix
# print(convert_pid_sid_to_eventkeyprefix('001', 's2017-06-18'))

def get_left_orientation_pid_sid(pid, sid):
    '''
    :param pid:
    :param sid: s2017-06-18
    :return:
    '''
    M_brush_with, M_floss_with, M_rins_with, M_ori_left, M_ori_right = get_event_METADATA()
    event_key_prefix = convert_pid_sid_to_eventkeyprefix(pid, sid)
    event_keys = [v for v in M_ori_left.keys() if v.startswith(event_key_prefix)]
    if len(event_keys) == 0:
        return 0
    return M_ori_left[event_keys[0]]

def get_right_orientation_pid_sid(pid, sid):
    '''
    :param pid:
    :param sid: s2017-06-18
    :return:
    '''
    M_brush_with, M_floss_with, M_rins_with, M_ori_left, M_ori_right = get_event_METADATA()
    event_key_prefix = convert_pid_sid_to_eventkeyprefix(pid, sid)
    event_keys = [v for v in M_ori_right.keys() if v.startswith(event_key_prefix)]
    if len(event_keys) == 0:
        return 0
    return M_ori_right[event_keys[0]]


def get_event_METADATA():

    D = pd.read_excel(metadata_filename, names=col_name)
    M_brush_with = dict()
    M_floss_with = dict()
    M_rins_with = dict()
    M_ori_left = dict()
    M_ori_right = dict()
    for i in range(len(D['video_name'])):
        key = D['video_name'].iloc[i]
        # print(key)
        is_normal_brush = int(D['is_normal_brush'].iloc[i])
        is_oralb_brush = int(D['is_oralb_brush'].iloc[i])
        is_flossing = int(D['is_flossing'].iloc[i])
        is_rinsing = int(D['is_rinsing'].iloc[i])
        left_ori = int(D['left_orientation'].iloc[i])
        right_ori = int(D['right_orientation'].iloc[i])

        if is_normal_brush == 1:
            M_brush_with[key] = NORMAL_BRUSHING
        elif is_oralb_brush == 1:
            M_brush_with[key] = ORALB_BRUSHING
        else:
            M_brush_with[key] = 'no'

        if is_flossing == 1:
            M_floss_with[key] = D['flossing_with'].iloc[i]
        else:
            M_floss_with[key] = 'no'

        if is_rinsing == 1:
            M_rins_with[key] = 'yes'
        else:
            M_rins_with[key] = 'no'
        M_ori_left[key] = left_ori
        M_ori_right[key] = right_ori
    return M_brush_with, M_floss_with, M_rins_with, M_ori_left, M_ori_right

def get_title_events(video_event_key):

    D = pd.read_excel(metadata_filename, names=col_name)
    M_brush = dict()
    M_floss = dict()
    M_rins = dict()
    fname = ''
    for i in range(len(D['video_name'])):
        key = D['video_name'].iloc[i]
        # print(key)
        is_normal_brush = int(D['is_normal_brush'].iloc[i])
        is_oralb_brush = int(D['is_oralb_brush'].iloc[i])
        is_flossing = int(D['is_flossing'].iloc[i])
        is_rinsing = int(D['is_rinsing'].iloc[i])

        if is_normal_brush == 1:
            M_brush[key] = 'normal_brush'
            fname += 'nb'
        elif is_oralb_brush == 1:
            M_brush[key] = 'oralB_brush'
            fname += 'ob'
        else:
            M_brush[key] = '-no'
            fname += '--'

        if is_flossing == 1:
            M_floss[key] = D['flossing_with'].iloc[i]
            fname += ( '_' +D['flossing_with'].iloc[i][:2])
        else:
            M_floss[key] = '-no'
            fname += '_--'

        if is_rinsing == 1:
            M_rins[key] = 'yes'
            fname += '_r'
        else:
            M_rins[key] = '-no'

    fname = M_brush[video_event_key][:1] + 'b_' + M_floss[video_event_key][:2] + '_' + M_rins[video_event_key][:1]

    ret = '(B=' + M_brush[video_event_key] + ' :: F=' + M_floss[video_event_key] + ' :: R=' + M_rins[
        video_event_key] + ')'

    # return ret, fname, M_brush[video_event_key], M_floss[video_event_key], M_rins[video_event_key]
    return ret, fname

