import matplotlib.pylab as plt
import pytz
import os

# input files
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

tz = pytz.timezone('US/Central')


BRUSHING = 'brushing'
FLOSSING = 'flossing'
MANUAL_BRUSHING = 'manual_brushing'
SMART_BRUSHING = 'smart_brushing'
STRING_FLOSSING = 'string_flossing'

LEFT_WRIST = 'left'
RIGHT_WRIST = 'right'
BOTH_WRIST = 'both'

columns_annotation_label = ['start_offset(sec)', 'end_offset(sec)', 'start_timestamp',
                            'end_timestamp', 'label']

columns_configuration_files = ['video_file_name', 'pid', 'wrist', 'is_manual_brushing',
                               'is_smart_brushing', 'is_flossing', 'flossing_with', 'is_rinsing',
                               'video_duration', 'is_video_paused', 'left_orientation',
                               'right_orientation', 'is_new_device']

def mkdir_if_not_exist(cur_dir):
    if not os.path.exists(cur_dir):
        os.mkdir(cur_dir)


data_dir = 'Y:/Data/ROBAS/FOR_OPEN_DATASET/mORAL_dataset_for_python/data/'

metadata_input_file = data_dir + 'annotation/Configurations.csv'

annotation_dir = data_dir + 'annotation/'
sensor_data_dir = data_dir + 'sensor_data/'
pos_event_input_data_dir = data_dir + 'sensor_data/only_positive_event_data/'
new_pos_event_input_data_dir = data_dir + 'sensor_data/new_only_positive_event_data/'
plot_dir  = data_dir + 'sensor_data/plots/'

feature_dir = data_dir + 'features_and_MLresults/'
# feature_dir = data_dir + 'features_and_MLresults_with90sec_bound/'

output_brushing_feature_dir = feature_dir + 'brushing_XY/'
# output_brushing_feature_dir = feature_dir + 'brushing_XY/'
output_flossing_feature_dir = feature_dir + 'flossing_XY/'

result_dir = output_brushing_feature_dir + 'MLresults/'

test_output_dir = feature_dir + 'Brushing_outputs/'

mkdir_if_not_exist(feature_dir)
mkdir_if_not_exist(output_brushing_feature_dir)
mkdir_if_not_exist(output_flossing_feature_dir)
mkdir_if_not_exist(result_dir)
mkdir_if_not_exist(test_output_dir)

