import os

from annotations.CONSTANTS import *
from utils.plot_signals import *
from annotations.read_annotations_and_configurations import *

# from annotations.META_DATA_video import *
# from annotations.read_annotations_and_configurations import get_event_configurations, get_orientations_and_device_info
from features.compute_featues_for_candidate_segment import generate_all_window_and_compute_brushing_features, \
    create_featurenames_for_brushing, create_featurenames_for_flossing, \
    generate_all_window_and_compute_flossing_features
from features.filter_based_on_time_duration import filter_based_on_duration, merge_candidates, combine_left_right, \
    filter_for_brushing
from input.import_stream_processor_inputs import *
from utils.annotations_gt_utils import *
from utils.file_utils import save_as_csv
from utils.generate_candidates import generate_candidates


# def get_brushing_gt_and_brushing_pred_timings_st_et(AGMO_r, cand_r, event_key):
#     if len(cand_r) == 0:
#         return 0, 0, 0, 0, 0
#
#     normal_brushing_times1, oralb_brushing_times, string_flossing_times, picks_flossing_times, rinsing_times, pause_times, video_time = get_annotations_eventwise(
#         event_key + '.xlsx')
#
#     nbst = min([v[0] / 1000.0 for v in normal_brushing_times1])
#     nbet = max([v[1] / 1000.0 for v in normal_brushing_times1])
#
#     dur_brushing_gt = nbet - nbst
#
#     for pt in pause_times:
#         paused_in_nbrushing = get_overlap_portion(nbst, nbet, pt[0] / 1000.0, pt[1] / 1000.0)
#         if paused_in_nbrushing > 10:
#             dur_brushing_gt -= paused_in_nbrushing
#
#     c_si = min([v[0] for v in cand_r])
#     c_ei = max([v[1] for v in cand_r])
#     c_st = AGMO_r[c_si][0]
#     c_et = AGMO_r[c_ei][0]
#
#     s_diff = nbst - c_st
#     e_diff = nbet - c_et
#
#     dur_brushing_pred = 0
#     for ct in cand_r:
#         st = AGMO_r[ct[0]][0]
#         et = AGMO_r[ct[1]][0]
#         # paused_dur = 0
#         # for pt in pause_times:
#         #     paused_in_nbrushing = get_overlap_portion(nbst, nbet, pt[0]/1000.0, pt[1]/1000.0)
#         #     paused_in_window = get_overlap_portion(st, et, pt[0]/1000.0, pt[1]/1000.0)
#         #     if (paused_in_nbrushing > 0) and paused_in_window > 0:
#         #         paused_dur += paused_in_window
#
#         dur_brushing_pred += (et - st)
#         # overlap_portion_nb = get_overlap_portion(st, et, nbst, nbet)
#         # # if overlap_portion_nb- paused_dur > 0:
#         # if overlap_portion_nb > 0:
#         #     dur_brushing_pred += overlap_portion_nb
#
#     return dur_brushing_gt, dur_brushing_pred, dur_brushing_gt - dur_brushing_pred, s_diff, e_diff


# def get_brushing_gt_and_brushing_pred_timings(AGMO_r, cand_r, event_key):
#     normal_brushing_times, oralb_brushing_times, string_flossing_times, picks_flossing_times, rinsing_times, pause_times, video_time = get_annotations_eventwise(
#         event_key + '.xlsx')
#
#     nbst = min([v[0] / 1000.0 for v in normal_brushing_times])
#     nbet = max([v[1] / 1000.0 for v in normal_brushing_times])
#     dur_brushing_gt = nbet - nbst
#
#     for pt in pause_times:
#         paused_in_nbrushing = get_overlap_portion(nbst, nbet, pt[0] / 1000.0, pt[1] / 1000.0)
#         if paused_in_nbrushing > 10:
#             dur_brushing_gt -= paused_in_nbrushing
#
#     dur_brushing_pred = 0
#     for ct in cand_r:
#         st = AGMO_r[ct[0]][0]
#         et = AGMO_r[ct[1]][0]
#         # paused_dur = 0
#         # for pt in pause_times:
#         #     paused_in_nbrushing = get_overlap_portion(nbst, nbet, pt[0]/1000.0, pt[1]/1000.0)
#         #     paused_in_window = get_overlap_portion(st, et, pt[0]/1000.0, pt[1]/1000.0)
#         #     if (paused_in_nbrushing > 0) and paused_in_window > 0:
#         #         paused_dur += paused_in_window
#
#         overlap_portion_nb = get_overlap_portion(st, et, nbst, nbet)
#         # if overlap_portion_nb- paused_dur > 0:
#         if overlap_portion_nb > 0:
#             dur_brushing_pred += overlap_portion_nb
#
#     return dur_brushing_gt, dur_brushing_pred, dur_brushing_gt - dur_brushing_pred


# def get_only_brushing_candidates(AGMO_r, cand_r, event_key):
#     normal_brushing_times = get_annotations_eventwise(event_key + '.xlsx')
#
#     nbst = min([v[0] / 1000.0 for v in normal_brushing_times])
#     nbet = max([v[1] / 1000.0 for v in normal_brushing_times])
#
#     cand_new = []
#     for ct in cand_r:
#         st = AGMO_r[ct[0]][0]
#         et = AGMO_r[ct[1]][0]
#
#         overlap_portion_nb = get_overlap_portion(st, et, nbst, nbet)
#         # if overlap_portion_nb- paused_dur > 0:
#         if overlap_portion_nb > 0:
#             cand_new.append(ct)
#
#     return cand_new


def do_proccess_positive_brushing_event_data_and_generate_featutes(pids):
    featurenames_brushing = create_featurenames_for_brushing()

    duration_outputs = []

    M_brush_with, M_floss_with, M_ori_left, M_ori_right, M_wrist, M_is_new_device = get_event_configurations()
    # pids = ['9a6b']
    for pid in pids:

        print('start')
        mbrushing_segmentList, sBrushing_segmentList, stringFlossing_segmentList = get_brushing_flossing_segments(pid)

        event_keys = [d for d in os.listdir(pos_event_input_data_dir) if
                      os.path.isdir(os.path.join(pos_event_input_data_dir, d)) and d.startswith(pid)]
        event_keys.sort()
        # print(pids)
        XX = []
        YY = []
        for event_key in event_keys:
            if 'plots' in event_key or event_key.endswith('20181202_084726'):
                continue
            if len(event_key) > 20:
                continue
            # if event_key in ['024_20181202_084726']:
            #     continue

            basedir = pos_event_input_data_dir + event_key + '/'
            # print('-----', basedir)
            if M_brush_with[event_key] != MANUAL_BRUSHING:
                continue

            annotation_filename = event_key + '.csv'
            AD = get_label_annotations(annotation_filename)
            if AD is None:
                print('---------ANNOTATION MISSING----', event_key)
                continue
            right_ori = M_ori_right[event_key]
            AGMO_r = get_accel_gyro_mag_orientation(basedir, RIGHT_WRIST, right_ori,
                                                    is_new_device=M_is_new_device[event_key])

            if len(AGMO_r) == 0:
                print('---------DATA MISSING----', event_key, '#right', len(AGMO_r))
                continue
            # print('>>>>', AD['start_timestamp'][0], AGMO_r[0][0])

            cand_r = generate_candidates(AGMO_r)
            # do_plotting(AGMO_r, AD, event_key, cand_r)
            cand_r = filter_for_brushing(cand_r, AGMO_r)
            # cand_r = get_only_brushing_candidates(AGMO_r, cand_r, event_key)

            do_plotting(AGMO_r, AD, event_key, cand_r)
            X_brush = generate_all_window_and_compute_brushing_features(pid, event_key, AGMO_r, cand_r)
            Y_brush = get_manual_brushing_labels(list(X_brush), mbrushing_segmentList)

            X_brush1 = [x for i, x in enumerate(X_brush) if Y_brush[i] == 1]
            Y_brush1 = [y for y in Y_brush if y == 1]

            output_filename = event_key + '_XY1.csv'
            if len(Y_brush1) > 0:
                duration_outputs.extend([[pid, event_key, x[4]] for x in X_brush1])
                print(event_key, '#ofPos', len(Y_brush1), len(Y_brush))
                XX.extend(X_brush1)
                YY.extend(Y_brush1)
                save_as_csv(X_brush1, Y_brush1, featurenames_brushing, output_brushing_feature_dir,
                            output_filename=output_filename)
            # do_plotting(AGMO_r, AGMO_l, AD, event_key, cand_l, cand_r, cands)
            else:
                print('--------------------NO candidate generated for ', event_key, len(Y_brush1), len(Y_brush))

        print('PID TOTAL ----->', pid, '#of windows', len(list(XX)), '#of only brushing',
              len([v for v in YY if v == 1]))
        if len(YY) > 0:
            save_as_csv(XX, YY, featurenames_brushing, output_brushing_feature_dir, output_filename=pid + '_XY1.csv')

    data = np.array(duration_outputs)
    pd_data = pd.DataFrame(data=data, columns=['pid', 'event_key', 'dur_brushing'])

    pd_data.to_csv(sensor_data_dir + 'brushing_candidate_durations.csv', encoding='utf-8', index=False)


def do_proccess_plot_candidates_for_positive_brushing_event(pids):
    duration_outputs = []

    M_brush_with, M_floss_with, M_ori_left, M_ori_right, M_wrist, M_is_new_device = get_event_configurations()
    # pids = ['9a6b']
    for pid in pids:

        print('start')
        event_keys = [d for d in os.listdir(pos_event_input_data_dir) if
                      os.path.isdir(os.path.join(pos_event_input_data_dir, d)) and d.startswith(pid)]
        event_keys.sort()
        # print(pids)
        for event_key in event_keys:
            if 'plots' in event_key or event_key.endswith('20181202_084726'):
                continue
            if len(event_key) > 20:
                continue
            # if event_key in ['024_20181202_084726']:
            #     continue

            # basedir = pos_event_input_data_dir + event_key + '/'
            basedir = new_pos_event_input_data_dir + event_key + '/'
            # print('-----', basedir)
            if M_brush_with[event_key] != MANUAL_BRUSHING:
                continue

            annotation_filename = event_key + '.csv'
            AD = get_label_annotations(annotation_filename)
            if AD is None:
                print('---------ANNOTATION MISSING----', event_key)
                continue
            right_ori = M_ori_right[event_key]
            AGMO_r = get_accel_gyro_mag_orientation(basedir, RIGHT_WRIST, right_ori,
                                                    is_new_device=M_is_new_device[event_key])

            if len(AGMO_r) == 0:
                print('---------DATA MISSING----', event_key, '#right', len(AGMO_r))
                continue
            # print('>>>>', AD['start_timestamp'][0], AGMO_r[0][0])

            cand_r = generate_candidates(AGMO_r)
            # do_plotting(AGMO_r, AD, event_key, cand_r)
            cand_r = filter_for_brushing(cand_r, AGMO_r)
            # cand_r = get_only_brushing_candidates(AGMO_r, cand_r, event_key)

            do_plotting(AGMO_r, AD, event_key, cand_r)


def process_pid_sid_negative_classes(pid, sid, mbrushing_segmentList, stringFlossing_segmentList):
    cur_dir = data_dir + pid + '/' + sid + '/'

    left_ori, right_ori, is_new_device = get_orientations_and_device_info(pid, sid)

    AGMO_r = get_accel_gyro_mag_orientation(cur_dir, RIGHT_WRIST, right_ori, is_new_device=is_new_device)
    AGMO_l = get_accel_gyro_mag_orientation(cur_dir, LEFT_WRIST, left_ori, is_new_device=is_new_device)

    if len(AGMO_l) == 0 or len(AGMO_r) == 0:
        print('---------DATA MISSING----', pid, sid, '#left', len(AGMO_l), '#right', len(AGMO_r))
        return [], [], [], []

    AGMO_l = aline_datastream(AGMO_l, [v[0] for v in AGMO_r])
    cand_l = generate_candidates(AGMO_l)
    cand_r = generate_candidates(AGMO_r)

    cand_l = filter_for_brushing(cand_l, AGMO_l)
    cand_r = filter_for_brushing(cand_r, AGMO_r)

    cands = combine_left_right(cand_l, cand_r)

    # if len(cand_l) > 0:
    #     X_left = generate_all_window_and_compute_brushing_features(pid, sid, AGMO_l, cand_l)
    #     X_df_left = pd.DataFrame(np.array(X_left), columns=featurenames)
    if len(cand_r) > 0:
        X_brush = generate_all_window_and_compute_brushing_features(pid, sid, AGMO_r, cand_r)
        Y_brush = get_manual_brushing_labels(list(X_brush), mbrushing_segmentList)
    else:
        X_brush, Y_brush = [], []

    if len(cands) > 0:
        X_floss = generate_all_window_and_compute_flossing_features(pid, sid, AGMO_l, AGMO_r, cands)
        Y_floss = get_flossing_labels(list(X_floss), stringFlossing_segmentList)
    else:
        X_floss, Y_floss = [], []

    return list(X_brush), list(Y_brush), list(X_floss), list(Y_floss)


def process_brushing_pid_sid(pid, sid, mbrushing_segmentList, sBrushing_segmentList=None):
    cur_dir = sensor_data_dir + pid + '/' + sid + '/'

    left_ori, right_ori, is_new_device = get_orientations_and_device_info(pid, sid)

    X_brush, Y_brush = [], []

    AGMO_r = get_accel_gyro_mag_orientation(cur_dir, RIGHT_WRIST, right_ori, is_new_device=is_new_device)
    if len(AGMO_r) > 0:
        cand_r = generate_candidates(AGMO_r)
        cand_r = filter_for_brushing(cand_r, AGMO_r)
        if len(cand_r) > 0:
            Xs = generate_all_window_and_compute_brushing_features(pid, sid, AGMO_r, cand_r, wrist=RIGHT_WRIST)
            Ys = get_manual_brushing_labels(list(Xs), mbrushing_segmentList, sBrushing_segmentList, wrist=RIGHT_WRIST)
            X_brush.extend(Xs)
            Y_brush.extend(Ys)
    else:
        print('---------DATA MISSING----', pid, sid, '#right', len(AGMO_r))
    AGMO_l = get_accel_gyro_mag_orientation(cur_dir, LEFT_WRIST, left_ori, is_new_device=is_new_device)
    if len(AGMO_l) > 0:
        cand_l = generate_candidates(AGMO_l)
        cand_l = filter_for_brushing(cand_l, AGMO_l)
        if len(cand_l) > 0:
            Xs = generate_all_window_and_compute_brushing_features(pid, sid, AGMO_l, cand_l, wrist=LEFT_WRIST)
            Ys = get_manual_brushing_labels(list(Xs), mbrushing_segmentList, sBrushing_segmentList, wrist=LEFT_WRIST)
            X_brush.extend(Xs)
            Y_brush.extend(Ys)
    else:
        print('---------DATA MISSING----', pid, sid, '#left', len(AGMO_l))
    print('Done for', pid, sid, 'Orientations:', left_ori, right_ori, is_new_device, '#pos:',
          len([v for v in Y_brush if v == 1]), 'total', len(Y_brush))
    return X_brush, Y_brush


def do_generate_feature_vector_and_labels_for_brushing(pids):

    featurenames_brushing = create_featurenames_for_brushing()
    # featurenames_flossing = create_featurenames_for_flossing()


    pids.sort()
    print(pids)
    for pid in pids:
        basedir = sensor_data_dir + pid + '/'
        print('----- Start for ', pid)
        days = [d for d in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, d)) and d.startswith('d')]
        days.sort()
        XXr, YYr, XXb, YYb = [], [], [], []
        mbrushing_segmentList, sBrushing_segmentList, stringFlossing_segmentList = get_brushing_flossing_segments(pid)
        # days = ['d2017-09-15']
        missed_brushed_days = []
        event_keys_pid = [v[3] for v in mbrushing_segmentList]
        brushed_days_groundtruth = [v[5:13] for v in event_keys_pid]
        for day in days:
            if day[1:].replace("-", "") not in brushed_days_groundtruth:
                continue
            X_brush, Y_brush = process_brushing_pid_sid(pid, day, mbrushing_segmentList, sBrushing_segmentList)
            if len(X_brush) > 0 and len([v for v in Y_brush if v == 1]) > 0:
                save_as_csv(X_brush, Y_brush, featurenames_brushing, output_brushing_feature_dir,
                            output_filename=pid + '_' + day + '_XY.csv')
                XXr.extend(X_brush)
                YYr.extend(Y_brush)
            else:
                missed_brushed_days.append(day)
                print('missed brushing day', day)
        print('Done for PID TOTAL ----->', pid, '#of windows', len(list(XXr)), '#of only brushing',
              len([v for v in YYr if v == 1]))
        print('EventKeys', event_keys_pid)
        print('brushing days', brushed_days_groundtruth, 'missed brushed_days', missed_brushed_days)
        if len(XXr) > 0:
            save_as_csv(XXr, YYr, featurenames_brushing, output_brushing_feature_dir, output_filename=pid + '_XY.csv')


if __name__ == "__main__":
    # do_proccess_for_all_events()
    pids = [d for d in os.listdir(sensor_data_dir) if os.path.isdir(os.path.join(sensor_data_dir, d)) and len(d) == 4]

    # pids.sort()
    print(pids)

    # do_proccess_plot_candidates_for_positive_brushing_event(pids)

    # do_proccess_positive_brushing_event_data_and_generate_featutes(pids)

    # do_proccess_negative_event_data_and_generate_featutes(pids)

    # generate both pos and neg
    do_generate_feature_vector_and_labels_for_brushing(pids)
