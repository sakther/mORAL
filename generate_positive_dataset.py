from annotations.CONSTANTS import *
from annotations.read_annotations_and_configurations import *
from features.filter_based_on_time_duration import filter_for_brushing
from input.import_stream_processor_inputs import *


# -------------import data-----------------------
from utils.generate_candidates import generate_candidates
from utils.plot_signals import do_plotting


def load_data_pd(filename):
    if not os.path.exists(filename):
        return None
    # data = pd.read_csv(filename, names=['ts', 'val'], index_col=False)
    data = pd.read_csv(filename, index_col=False)
    return data


def get_data(data_dir):
    axL = load_data_pd(os.path.join(data_dir, ax_left_filename))
    ayL = load_data_pd(os.path.join(data_dir, ay_left_filename))
    azL = load_data_pd(os.path.join(data_dir, az_left_filename))
    gxL = load_data_pd(os.path.join(data_dir, gx_left_filename))
    gyL = load_data_pd(os.path.join(data_dir, gy_left_filename))
    gzL = load_data_pd(os.path.join(data_dir, gz_left_filename))

    axR = load_data_pd(os.path.join(data_dir, ax_right_filename))
    ayR = load_data_pd(os.path.join(data_dir, ay_right_filename))
    azR = load_data_pd(os.path.join(data_dir, az_right_filename))
    gxR = load_data_pd(os.path.join(data_dir, gx_right_filename))
    gyR = load_data_pd(os.path.join(data_dir, gy_right_filename))
    gzR = load_data_pd(os.path.join(data_dir, gz_right_filename))

    return axL, ayL, azL, gxL, gyL, gzL, axR, ayR, azR, gxR, gyR, gzR


def save_data_as_csv(data, filename):
    if data is not None:
        data.to_csv(filename, index=False)


# -------------import data-----------------------

def get_video_start_end_times(pid):
    annotation_files = [d for d in os.listdir(annotation_dir) if d.startswith(pid)]
    M = {}
    for annotation_file in annotation_files:

        if '_ext' in annotation_file:
            continue
        event_key = annotation_file[:-4]

        D = get_label_annotations(annotation_file)
        start_time = -1
        end_time = -1
        is_mbrush, is_sbrush, is_flossing = False, False, False
        for i in range(len(D['label'])):
            st = D['start_timestamp'].iloc[i]
            et = D['end_timestamp'].iloc[i]
            lb = D['label'].iloc[i]

            if start_time == -1:
                start_time = st
            else:
                start_time = min(start_time, st)
            if end_time == -1:
                end_time = et
            else:
                end_time = max(end_time, et)
            if lb == MANUAL_BRUSHING:
                is_mbrush = True
            if lb == SMART_BRUSHING:
                is_sbrush = True
            if lb == STRING_FLOSSING:
                is_flossing = True

        M[event_key] = [start_time - 120, end_time + 120, is_mbrush, is_sbrush, is_flossing]
    return M


def get_subArray(data, st, et):
    if data is None:
        return None

    data_tmp = data.copy()
    data_tmp = data_tmp[data_tmp['ts'] >= st]
    data_tmp = data_tmp[data_tmp['ts'] <= et]
    return data_tmp


import matplotlib

font = {'family': 'normal',
        'weight': 'bold',
        'size': 20}

matplotlib.rc('font', **font)
# matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('axes', labelsize=20)


def do_plots(axLt, ayLt, azLt, gxLt, gyLt, gzLt, axRt, ayRt, azRt, gxRt, gyRt, gzRt, cur_dir):
    try:
        plt.plot(axLt['ts'], axLt['val'], 'r-', label='AxL')
        plt.plot(axLt['ts'], [0 for _ in axLt['val']], '--k')
        plt.plot(ayLt['ts'], ayLt['val'] - 2, 'g-', label='AyL')
        plt.plot(axLt['ts'], [-2 for _ in axLt['val']], '--k')
        plt.plot(azLt['ts'], azLt['val'] - 4, 'b-', label='AzL')
        plt.plot(axLt['ts'], [-4 for _ in axLt['val']], '--k')

        plt.plot(axRt['ts'], axRt['val'] - 8, 'r-', label='AxR')
        plt.plot(axLt['ts'], [-8 for _ in axLt['val']], '--k')
        plt.plot(ayRt['ts'], ayRt['val'] - 10, 'g-', label='AyR')
        plt.plot(axLt['ts'], [-10 for _ in axLt['val']], '--k')
        plt.plot(azRt['ts'], azRt['val'] - 12, 'b-', label='AzR')
        plt.plot(axLt['ts'], [-12 for _ in axLt['val']], '--k')
        plt.legend()
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(15, 10)
        plt.savefig(cur_dir + 'Accel.png')
        plt.savefig(cur_dir + 'Accel.pdf')
        plt.close()

        plt.plot(gxLt['ts'], gxLt['val'], 'r-', label='GxL')
        plt.plot(gyLt['ts'], gyLt['val'] - 100, 'g-', label='GyL')
        plt.plot(gzLt['ts'], gzLt['val'] - 200, 'b-', label='GzL')

        plt.plot(gxRt['ts'], gxRt['val'] - 400, 'r-', label='GxR')
        plt.plot(gyRt['ts'], gyRt['val'] - 500, 'g-', label='GyR')
        plt.plot(gzRt['ts'], gzRt['val'] - 600, 'b-', label='GzR')
        plt.legend()
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(15, 10)
        plt.savefig(cur_dir + 'Gyro.png')
        plt.savefig(cur_dir + 'Gyro.pdf')
        plt.close()
    except:
        pass


def generate_positive_event_data_from_video_timing(pids):
    for pid in pids:
        basedir = sensor_data_dir + pid + '/'
        M = get_video_start_end_times(pid)
        expected_event_keys = list(M.keys())
        found_event_keys = []
        print('----- Start for ', pid)
        print('Events:', list(M.keys()))
        days = [d for d in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, d)) and d.startswith('d')]
        days.sort()
        for day in days:
            cur_dir = sensor_data_dir + pid + '/' + day + '/'
            axL, ayL, azL, gxL, gyL, gzL, axR, ayR, azR, gxR, gyR, gzR = get_data(cur_dir)
            if axL is not None and axR is not None:
                day_st, day_et = min(axL.iloc[0, 0], axR.iloc[0, 0]), max(axL.iloc[-1, 0], axR.iloc[-1, 0])
            elif axL is not None:
                day_st, day_et = axL.iloc[0, 0], axL.iloc[-1, 0]
            elif axR is not None:
                day_st, day_et = axR.iloc[0, 0], axR.iloc[-1, 0]
            else:
                continue

            for event_key, st_et in M.items():
                st = st_et[0]
                et = st_et[1]

                if st >= day_st and et <= day_et:
                    cur_output_dir = new_pos_event_input_data_dir + str(event_key) + '/'
                    if not os.path.exists(cur_output_dir):
                        os.mkdir(cur_output_dir)
                    axLt = get_subArray(axL, st, et)
                    ayLt = get_subArray(ayL, st, et)
                    azLt = get_subArray(azL, st, et)
                    gxLt = get_subArray(gxL, st, et)
                    gyLt = get_subArray(gyL, st, et)
                    gzLt = get_subArray(gzL, st, et)

                    axRt = get_subArray(axR, st, et)
                    ayRt = get_subArray(ayR, st, et)
                    azRt = get_subArray(azR, st, et)
                    gxRt = get_subArray(gxR, st, et)
                    gyRt = get_subArray(gyR, st, et)
                    gzRt = get_subArray(gzR, st, et)
                    save_data_as_csv(axLt, os.path.join(cur_output_dir, ax_left_filename))
                    save_data_as_csv(ayLt, os.path.join(cur_output_dir, ay_left_filename))
                    save_data_as_csv(azLt, os.path.join(cur_output_dir, az_left_filename))
                    save_data_as_csv(gxLt, os.path.join(cur_output_dir, gx_left_filename))
                    save_data_as_csv(gyLt, os.path.join(cur_output_dir, gy_left_filename))
                    save_data_as_csv(gzLt, os.path.join(cur_output_dir, gz_left_filename))

                    save_data_as_csv(axRt, os.path.join(cur_output_dir, ax_right_filename))
                    save_data_as_csv(ayRt, os.path.join(cur_output_dir, ay_right_filename))
                    save_data_as_csv(azRt, os.path.join(cur_output_dir, az_right_filename))
                    save_data_as_csv(gxRt, os.path.join(cur_output_dir, gx_right_filename))
                    save_data_as_csv(gyRt, os.path.join(cur_output_dir, gy_right_filename))
                    save_data_as_csv(gzRt, os.path.join(cur_output_dir, gz_right_filename))
                    do_plots(axLt, ayLt, azLt, gxLt, gyLt, gzLt, axRt, ayRt, azRt, gxRt, gyRt, gzRt, cur_output_dir)
                    found_event_keys.append(event_key)
                    # print('Done for', event_key)
        not_found_event_keys = [v for v in expected_event_keys if v not in found_event_keys]
        print('Not found for', pid, not_found_event_keys)


def plot_positive_events(pids):
    for pid in pids:
        M = get_video_start_end_times(pid)
        print('----- Start for ', pid)
        print('Events:', list(M.keys()))
        # days = [d for d in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, d)) and d.startswith('d')]

        for event_key, st_et in M.items():
            cur_output_dir = new_pos_event_input_data_dir + str(event_key) + '/'
            axL, ayL, azL, gxL, gyL, gzL, axR, ayR, azR, gxR, gyR, gzR = get_data(cur_output_dir)

            do_plots(axL, ayL, azL, gxL, gyL, gzL, axR, ayR, azR, gxR, gyR, gzR, cur_output_dir)


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

            do_plotting(AGMO_r, AD, event_key, cand_r, output_dir=basedir)



# Not found for 999e ['999e_20180629_23591', '999e_20180701_23580']
if __name__ == "__main__":
    # do_proccess_for_all_events()
    pids = [d for d in os.listdir(sensor_data_dir) if os.path.isdir(os.path.join(sensor_data_dir, d)) and len(d) == 4]
    # pids = ['820c', '9a6b', 'b67b']
    # pids = ['b67b']
    # pids.sort()
    print(pids)
    generate_positive_event_data_from_video_timing(pids)
    # plot_positive_events(pids)
    do_proccess_plot_candidates_for_positive_brushing_event(pids)
