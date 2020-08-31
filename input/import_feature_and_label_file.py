import pandas as pd
import numpy as np
from features.compute_featues_for_candidate_segment import *
from annotations.CONSTANTS import *

def get_XY(pid):
    XYdf = pd.read_csv(output_brushing_feature_dir + pid + '_XY.csv', index_col=False)
    XYdf = XYdf[XYdf['label']>= 0]
    feature_names = list(XYdf.columns)
    # XY = list(XY.values)
    XY = XYdf.values[:,3:].astype(float)
    XY = XY[~np.isnan(XY).any(axis=1)]
    Y = XY[:,-1].astype(int)

    T = XY[:, 0]
    X = XY[:, 2:-1]
    Y = XY[:,-1].astype(int)
    feature_names = feature_names[5:-1]

    return feature_names, T, X, Y