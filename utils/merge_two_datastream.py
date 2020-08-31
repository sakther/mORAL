import numpy as np

def getInterpoletedValue(g0, g1, t0, t1, t):
    g = g0 + (g1 - g0) * (t - t0) / (t1 - t0)
    return g

def merge2stream(At, Gt, Gx):
    i = 0
    j = 0
    _Gx = [0] * len(At)
    while (i < len(At)) and (j < len(Gt)):
        while Gt[j] < At[i]:
            j = j + 1
            if j >= len(Gt):
                break
        if j < len(Gt):
            if (At[i] == Gt[j]) | (j == 0):
                _Gx[i] = Gx[j]
            else:
                _Gx[i] = getInterpoletedValue(Gx[j - 1], Gx[j], Gt[j - 1], Gt[j], At[i])
        i = i + 1
    return _Gx

def merge(At, Ax, Ay, Az, Gt, Gx, Gy, Gz):
    i = 0
    j = 0
    _Gx = [0] * len(At)
    _Gy = [0] * len(At)
    _Gz = [0] * len(At)
    while (i < len(At)) and (j < len(Gt)):
        #         print(Gt[j])
        #         print(At[j])
        while Gt[j] < At[i]:
            j = j + 1
            if j >= len(Gt):
                break
        if j < len(Gt):
            if (At[i] == Gt[j]) | (j == 0):
                _Gx[i] = Gx[j]
                _Gy[i] = Gy[j]
                _Gz[i] = Gz[j]
            else:
                _Gx[i] = getInterpoletedValue(Gx[j - 1], Gx[j], Gt[j - 1], Gt[j], At[i])
                _Gy[i] = getInterpoletedValue(Gy[j - 1], Gy[j], Gt[j - 1], Gt[j], At[i])
                _Gz[i] = getInterpoletedValue(Gz[j - 1], Gz[j], Gt[j - 1], Gt[j], At[i])
        i = i + 1
    return At, Ax, Ay, Az, _Gx, _Gy, _Gz


def merge_accel_gyro(accel, gyro):
    A = np.array(accel)
    G = np.array(gyro)
    At, Ax, Ay, Az, _Gx, _Gy, _Gz = merge(A[:,0], A[:,1], A[:,2], A[:,3], G[:,0], G[:,1], G[:,2], G[:,3])
    AG = list(np.column_stack((At, Ax, Ay, Az, _Gx, _Gy, _Gz)))
    return AG

def aline_datastream(data, t):
    '''

    :param data: each row is (ts, d1, d2, d3, ...); here ts is timestamp, d1 is the first data and so on
    :param t: t is the list of timestamp
    :return: converted data of size |t| where first column is t
    '''

    n = len(data[0])
    data = np.array(data)
    t_old = data[:,0]
    new_data = np.array(t)
    for i in range(1, n):
        d_tmp = merge2stream(t, t_old, data[:, i])
        new_data = np.column_stack((new_data, np.array(d_tmp)))

#     print('|t|=', len(t), '|old data|', len(list(data)), '|new data|', len(list(new_data)))
    return new_data
