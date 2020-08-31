from typing import List
import os
import numpy as np
import pytz
tz = pytz.timezone('US/Central')

LEFT_WRIST = 'leftwrist'
RIGHT_WRIST = 'rightwrist'

ax_left_filename = 'left-wrist-accelx.csv'
ay_left_filename = 'left-wrist-accely.csv'
az_left_filename = 'left-wrist-accelz.csv'
gx_left_filename = 'left-wrist-gyrox.csv'
gy_left_filename = 'left-wrist-gyroy.csv'
gz_left_filename = 'left-wrist-gyroz.csv'
ax_right_filename = 'right-wrist-accelx.csv'
ay_right_filename = 'right-wrist-accely.csv'
az_right_filename = 'right-wrist-accelz.csv'
gx_right_filename = 'right-wrist-gyrox.csv'
gy_right_filename = 'right-wrist-gyroy.csv'
gz_right_filename = 'right-wrist-gyroz.csv'

def get_fileName(cur_dir, file_sufix):
    filenames = [name for name in os.listdir(cur_dir) if
                 name.endswith(file_sufix)]
    if len(filenames) > 0:
        return filenames[0]
    else:
        return None

def convert_sample(sample):
    return list([float(x.strip()) for x in sample.split(',')])


def line_parser(input):
    ts, offset, sample = input.split(',', 2)

    start_time = int(float(ts))
    dp = [start_time]
    sample = convert_sample(sample)
    dp.extend(sample)

    return dp

def load_data(filename):
    fp = open(filename)
    file_content = fp.read()
    fp.close()

    lines = file_content.splitlines()
    data = list(map(line_parser, lines))

    return data

def load_datapointarray(data_dir, filename):
    data = load_data(data_dir + get_fileName(data_dir, filename + '.csv'))
    return data

ACCEL_LEFT = 'ACCEL_LEFT'
ACCEL_RIGHT = 'ACCEL_RIGHT'
GYRO_LEFT = 'GYRO_LEFT'
GYRO_RIGHT = 'GYRO_RIGHT'

def append_to_file(filename, txt):
    fh = open(filename, 'a')
    fh.write(txt + '\n')
    fh.close()

def write_matrix(data, filename):
    data = np.array(data)
    np.savetxt(filename, data, fmt='%s', delimiter=",")

def export_datastream(filename: str, data):

    write_matrix(data, filename)

    # for dp in data:
    #     txt = str(dp[0]) + ',' + str(dp[1])
    #     append_to_file(filename, txt)

# data is a list of dp=(t, x, y, z)
def export_as_streamprocessor(data_dir, data, data_type):
    t = [v[0] for v in data]
    x = [[v[0], v[1]] for v in data]
    y = [[v[0], v[2]] for v in data]
    z = [[v[0], v[3]] for v in data]

    if data_type == ACCEL_LEFT:
        export_datastream(data_dir + ax_left_filename, x)
        export_datastream(data_dir + ay_left_filename, y)
        export_datastream(data_dir + az_left_filename, z)
    if data_type == ACCEL_RIGHT:
        export_datastream(data_dir + ax_right_filename, x)
        export_datastream(data_dir + ay_right_filename, y)
        export_datastream(data_dir + az_right_filename, z)

    if data_type == GYRO_LEFT:
        export_datastream(data_dir + gx_left_filename, x)
        export_datastream(data_dir + gy_left_filename, y)
        export_datastream(data_dir + gz_left_filename, z)
    if data_type == GYRO_RIGHT:
        export_datastream(data_dir + gx_right_filename, x)
        export_datastream(data_dir + gy_right_filename, y)
        export_datastream(data_dir + gz_right_filename, z)
