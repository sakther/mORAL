from utils.smooth_output import do_smooth_output
from utils.util_import import *

def evaluate_single_random_forest(X_tr, Y_tr, X_ts, Y_ts, T_ts, n_estimators=100):

    w0 = len([v for v in Y_tr if v == 0])/len(list(Y_tr))
    w1 = len([v for v in Y_tr if v == 1])/len(list(Y_tr))
    # clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=n_estimators, class_weight={0:w0, 1:w1})
    clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=n_estimators, class_weight={0:w1, 1:w0})
    clf.fit(X_tr, Y_tr)

    Y_ts_pred = np.array(clf.predict(X_ts))
    return Y_ts_pred

def evaluate_single_random_forest_smooth(X_tr, Y_tr, X_ts, Y_ts, T_ts, n_estimators=100):

    Y_ts_pred = evaluate_single_random_forest(X_tr, Y_tr, X_ts, Y_ts, T_ts, n_estimators)
    Y_ts_pred_smooth = do_smooth_output(T_ts, Y_ts_pred)
    return Y_ts_pred_smooth


