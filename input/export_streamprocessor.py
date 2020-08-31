from typing import List
# from domain.datapoint import DataPoint
import numpy as np

def append_to_file(filename, txt):
    fh = open(filename, 'a')
    fh.write(txt + '\n')
    fh.close()

# def export_datastream(filename: str, data: List[DataPoint]):
#
#     for dp in data:
#         txt = dp.start_time + ',' + dp.start_time.timestamp() + ',' + dp.sample
#         append_to_file(filename, txt)
#
# def export_datastream_ascloud(filename: str, data: List[DataPoint]):
#
#     for dp in data:
#         sample_str = str(dp.sample[0])
#         for i in range(1, len(dp.sample)):
#             sample_str += str(dp.sample[i])
#
#         txt = str(dp.start_time.timestamp()) + ',' + str(dp.offset) + ',' + sample_str
#         append_to_file(filename, txt)
#
#
