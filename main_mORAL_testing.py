import os

from annotations.CONSTANTS import *
from main_LOSOCV_brushing import get_recal_precision_f1_accuracy_CM
from models.AdaBoost import run_adaboost
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


def get_brushing_features_and_labels_for_pid_daywise(pid, sid, mbrushing_segmentList, sBrushing_segmentList=None):
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


def get_brushing_features_for_pid_day(pid, sid):
    cur_dir = sensor_data_dir + pid + '/' + sid + '/'

    left_ori, right_ori, is_new_device = get_orientations_and_device_info(pid, sid)

    X_brush = [], []

    AGMO_r = get_accel_gyro_mag_orientation(cur_dir, RIGHT_WRIST, right_ori, is_new_device=is_new_device)
    if len(AGMO_r) > 0:
        cand_r = generate_candidates(AGMO_r)
        cand_r = filter_for_brushing(cand_r, AGMO_r)
        if len(cand_r) > 0:
            Xs = generate_all_window_and_compute_brushing_features(pid, sid, AGMO_r, cand_r, wrist=RIGHT_WRIST)
            X_brush.extend(Xs)
    else:
        print('---------DATA MISSING----', pid, sid, '#right', len(AGMO_r))
    AGMO_l = get_accel_gyro_mag_orientation(cur_dir, LEFT_WRIST, left_ori, is_new_device=is_new_device)
    if len(AGMO_l) > 0:
        cand_l = generate_candidates(AGMO_l)
        cand_l = filter_for_brushing(cand_l, AGMO_l)
        if len(cand_l) > 0:
            Xs = generate_all_window_and_compute_brushing_features(pid, sid, AGMO_l, cand_l, wrist=LEFT_WRIST)
            X_brush.extend(Xs)
    else:
        print('---------DATA MISSING----', pid, sid, '#left', len(AGMO_l))
    print('Done for', pid, sid, 'total', len(X_brush))
    return X_brush


def get_only_marked_manual_brushing_days(pid):
    basedir = sensor_data_dir + pid + '/'
    days = [d for d in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, d)) and d.startswith('d')]
    days.sort()

    mBrushing_eventkeys = get_manual_brushing_annotation_eventkeys(pid)
    brushed_days_groundtruth = [v[5:13] for v in mBrushing_eventkeys]

    mBrushing_days = [day for day in days if day[1:].replace("-", "") in brushed_days_groundtruth]
    mBrushing_days.sort()

    return mBrushing_days


def process_mORAL_and_validate(pids):
    pids.sort()
    print(pids)
    for pid in pids:
        print('----- Start for ', pid)
        mBrushing_days = get_only_marked_manual_brushing_days(pid)
        Xs, Ys_true, Ys_pred = [], [], []

        mbrushing_segmentList, sBrushing_segmentList, stringFlossing_segmentList = get_brushing_flossing_segments(pid)

        for day in mBrushing_days:
            X_brush, Y_brush = get_brushing_features_and_labels_for_pid_daywise(pid, day, mbrushing_segmentList,
                                                                                sBrushing_segmentList)
            if len(X_brush) > 0:
                Xday = np.array(X_brush)
                Y_preds = run_adaboost(Xday[:, 5:])

                Xs.extend(X_brush)
                Ys_true.extend(Y_brush)
                Ys_pred.extend(Y_preds)
        Ys_pred = [v for i, v in enumerate(Ys_pred) if Ys_true[i]>=0]
        Ys_true = [v for i, v in enumerate(Ys_true) if Ys_true[i]>=0]
        print('>>>>>> Result for', pid, get_recal_precision_f1_accuracy_CM(Ys_true, Ys_pred))


def process_mORAL(pids):
    pids.sort()
    print(pids)
    detected_brushing_events = []
    for pid in pids:
        print('----- Start for ', pid)
        mBrushing_days = get_only_marked_manual_brushing_days(pid)
        Xs, Ys_preds = [], []

        for day in mBrushing_days:
            X_brush = get_brushing_features_for_pid_day(pid, day)
            if len(X_brush) > 0:
                Y_preds = run_adaboost(X_brush[5:])
                Xs.extend(X_brush)
                Ys_preds.extend(Y_preds)

        for i, x in enumerate(Xs):
            if Ys_preds[i] == 1:
                pid, day, wrist, stime, etime = x[0], x[1], x[2], x[3], x[4]
                detected_brushing_events.append([pid, day, wrist, stime, etime])
    pd_data = pd.DataFrame(data=np.array(detected_brushing_events),columns=['pid', 'day', 'wrist', 'stime', 'etime'])
    pd_data.to_csv(test_output_dir + 'brushing_events.csv', encoding='utf-8', index=False)

    return detected_brushing_events


if __name__ == "__main__":
    pids = [d for d in os.listdir(sensor_data_dir) if os.path.isdir(os.path.join(sensor_data_dir, d)) and len(d) == 4]
    print(pids)

    process_mORAL_and_validate(pids)
    # process_mORAL(pids)
