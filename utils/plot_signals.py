from annotations.CONSTANTS import *

def plot_only_accel_signals(data, pltax, wrist):
    if len(data) == 0:
        return
    t =  [v[0] for v in data]
    ax =  [v[1] for v in data]
    ay =  [v[2]-4 for v in data]
    az =  [v[3]-8 for v in data]
    pltax.plot([t[0], t[-1]], [0, 0], '--k')
    pltax.plot([t[0], t[-1]], [-4, -4], '--k')
    pltax.plot([t[0], t[-1]], [-8, -8], '--k')

    if wrist == RIGHT_WRIST:
        pltax.plot(t, ax, '-b', label = 'Ax-'+wrist[:2])
        pltax.plot(t, ay, '-r', label = 'Ay-'+wrist[:2])
        pltax.plot(t, az, '-g', label = 'Az-'+wrist[:2])
    else:
        pltax.plot(t, ax, '-m', label = 'Ax-'+wrist[:2])
        pltax.plot(t, ay, '-c', label = 'Ay-'+wrist[:2])
        pltax.plot(t, az, '-y', label = 'Az-'+wrist[:2])
    #plt.plot(t, am, '-k', label = 'Am')
    return pltax

def plot_annotation(D, bottom, top, pltax, start_timestamp=0):

    # col_name = ['start_timestamp', 'end_timestamp', 'label']
    for i in range(len(D['start_timestamp'])):
        # print(str(D['start_timestamp'].iloc[i]) + ', ' + str(D['end_timestamp'].iloc[i]) + ', ' + str(D['label'].iloc[i]))
        st = D['start_timestamp'].iloc[i]/1000.0
        et = D['end_timestamp'].iloc[i]/1000.0
        lb = D['label'].iloc[i]
        lWidth = 3
        if lb == MANUAL_BRUSHING:
            pltax.text(st, top-bottom/200, 'BR', fontsize=25)
            pltax.axvspan(st, et, ymin=bottom, ymax=top, alpha=0.2, color='c', label='BR')
        if lb == STRING_FLOSSING:
            pltax.text(st, top-bottom/200, 'FL', fontsize=30)
            pltax.axvspan(st, et, ymin=bottom, ymax=top, alpha=0.2, color='green', label='FL')
    return pltax

def plot_candidate_windows(t, cand, pltax, offset=10, wrist=RIGHT_WRIST):
    for v in cand:
        if wrist == LEFT_WRIST:
            pltax.plot([t[v[0]], t[v[1]]], [offset, offset+0.3], '-r')
        elif wrist == RIGHT_WRIST:
            pltax.plot([t[v[0]], t[v[1]]], [offset, offset+0.3], '-b')
        else:
            pltax.plot([t[v[0]], t[v[1]]], [offset, offset-0.3], '-g')
    return pltax

def do_plotting(AGMO_r, AD, event_key, cand_r, output_dir = ''):
    '''
    :param AGMO: (t, ax, ay, az, gx, gy, gz, amag, gmag, roll, pitch, yaw)
    :param AD: annotation data
    :return:
    '''
    if output_dir == '':
        output_dir = plot_dir

    f, ax = plt.subplots()
    # ax = plt.figure()
    ax.set_title(event_key)
    pltax = plot_annotation(AD, -20, 2, ax)
    pltax = plot_only_accel_signals(AGMO_r, pltax, RIGHT_WRIST)

    t =  [v[0] for v in AGMO_r]
    pltax = plot_candidate_windows(t, cand_r, pltax, 4, RIGHT_WRIST)

    plt.legend()
    # plt.show()
    print('saved', output_dir + event_key + '_brushing.png')
    plt.savefig(output_dir + event_key + '_brushing.png')
    plt.close()
