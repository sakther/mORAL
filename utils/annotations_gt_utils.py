# import pandas as pd
# import matplotlib.pylab as plt
# import math
# import statistics
# import numpy as np
# import os
# import datetime
# import time
# from pytz import timezone
#
# from annotations.META_DATA_video import get_event_METADATA
#
# SURFACE_SWITCH = 'surface-switch'
# BRUSHING = 'brushing'
# PAUSE = 'pause'
# RINSING = 'rinsing'
# FLOSSING = 'flossing'
#
# NORMAL_BRUSHING = 'normal_brush'
# ORALB_BRUSHING = 'oralB_brush'
# STRING_FLOSSING = 'string'
#
#
#
# def get_duration_map():
#     col_name = ['video_name', 'start_timestamp', 'end_timestamp', 'duration']
#     D = pd.read_csv(video_start_end_filename, names=col_name)
#     M = dict()
#     for i in range(len(D['video_name'])):
#         key = D['video_name'].iloc[i]
#         value = [D['start_timestamp'].iloc[i], D['end_timestamp'].iloc[i], D['duration'].iloc[i]]
#         M[key] = value
#
#     return M
# def get_number_of_seconds(st):
#     if st == 'nan':
#         return -1
#     mins = int(st.split(':')[0])
#     if mins == 12:
#         mins = 0
#     sec = int(st.split(':')[1])
#     return 60 * mins + sec
#
# def     get_annotations(annotation_filename):
#     if os.path.exists(annotation_dir + annotation_filename):
#         col_name = ['start_timestamp', 'end_timestamp', 'label']
#         # print(annotation_dir + annotation_filename)
#         D = pd.read_excel(annotation_dir + annotation_filename, names=col_name)
#         return D
#     else:
#         return None
#
# # -----------------------------------------------------------------------------
#
#
# def get_event_time_intervals(annotation_filename, v_stimestamp):
#     col_name = ['start_videotime', 'end_videotime', 'label']
#     D = pd.read_excel(annotation_dir + annotation_filename, names = col_name)
#     D['start_timestamp'] = [v_stimestamp + 1000*get_number_of_seconds(str(v)) for v in D['start_videotime']]
#     D['end_timestamp'] = [v_stimestamp + 1000*get_number_of_seconds(str(v)) for v in D['end_videotime']]
#     D['duration'] = D['end_timestamp'] - D['start_timestamp']
#     return D
#
# def get_annotations_eventwise(annotation_filename):
#     M_st_et_dur = get_duration_map()
#     event_key = annotation_filename[:-5]
#     M_brush_with, M_floss_with, M_rins_with, M_ori_left, M_ori_right = get_event_METADATA()
#
#     video_st = M_st_et_dur[event_key][0]
#     video_et = M_st_et_dur[event_key][1]
#
#     D = get_event_time_intervals(annotation_filename, video_st)
#
#     string_flossing_times = []
#     picks_flossing_times = []
#     normal_brushing_times = []
#     oralb_brushing_times = []
#     rinsing_times = []
#     pause_times = []
#     video_time = [[video_st, video_et]]
#
#     for i in range(len(D['start_timestamp'])):
#         st = D['start_timestamp'].iloc[i]
#         et = D['end_timestamp'].iloc[i]
#         lb = D['label'].iloc[i]
#         if lb == BRUSHING:
#             if M_brush_with[event_key] == NORMAL_BRUSHING:
#                 normal_brushing_times.append([st, et])
#             else:
#                 oralb_brushing_times.append([st, et])
#         if lb == FLOSSING:
#             if M_floss_with[event_key] == STRING_FLOSSING:
#                 string_flossing_times.append([st, et])
#             else:
#                 picks_flossing_times.append([st, et])
#         if lb == RINSING:
#             rinsing_times.append([st, et])
#         if lb == PAUSE:
#             pause_times.append([st, et])
#     return normal_brushing_times, oralb_brushing_times, string_flossing_times, picks_flossing_times, rinsing_times, pause_times, video_time
#
# def get_annotation_for_participant(pid):
#     event_keys = [d for d in os.listdir(annotation_dir) if d.startswith(pid)]
#     M=dict()
#     for event_key in event_keys:
#         # if event_key in ['013_20180624_214423.xlsx', '014_20180703_003652.xlsx', '016_20180708_074835.xlsx', '016_20180711_012753.xlsx', '017_20180711_005431.xlsx']:
#         #     continue
#         if '_ext' in event_key:
#             continue
#         M[event_key] = get_annotations_eventwise(event_key)
#
#     return M
#
# def get_video_timings_for_pid(pid):
#     event_keys = [d for d in os.listdir(annotation_dir) if d.startswith(pid)]
#     video_timings = []
#
#     for event_key in event_keys:
#         normal_brushing_times, oralb_brushing_times, string_flossing_times, picks_flossing_times, rinsing_times, pause_times, video_time = get_annotations_eventwise(event_key)
#         video_timings.append(video_time[0])
#
#     return video_timings
#
