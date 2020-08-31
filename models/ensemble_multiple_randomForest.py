from utils.smooth_output import do_smooth_output
from utils.util_import import *

N_tree = 50

def evaluate_ensemble_multiple_random_forest(X_tr, Y_tr, X_ts, Y_ts, T_ts):

    X_tr = np.array(X_tr)
    Y_tr = np.array(Y_tr)

    X_tr0 = list(X_tr[Y_tr == 0])
    X_tr1 = list(X_tr[Y_tr == 1])
    len0 = len(X_tr0)
    len1 = min(len(X_tr1)*7, len(X_tr0))

    np.random.shuffle(X_tr0)
    # X_tr0 = list(np.random.shuffle(X_tr0))
    nSample = N_tree * len1
    # X_tr0 = X_tr0[:nSample]

    RFClfs = []

    for i in range(N_tree):
        clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=100)
        X =[v for v in X_tr1]
        np.random.shuffle(X_tr0)
        X.extend([v for v in X_tr0[:len1]])

        Y = [1 for v in X_tr1]
        Y.extend([0]*len1)
        clf.fit(X, Y)
        RFClfs.append(clf)

    Ys = []
    for clf in RFClfs:
        Y_ts_preds = np.array(clf.predict(X_ts))
        Ys.append(Y_ts_preds)

    Y_ts_pred = [0]*len(Y_ts)
    for i in range(len(Y_ts)):
        mn = np.mean([v[i] for v in Ys])
        if mn >= 0.5:
            Y_ts_pred[i] = 1

    return Y_ts_pred


def evaluate_ensemble_multiple_random_forest_smoothed(X_tr, Y_tr, X_ts, Y_ts, T_ts):

    Y_ts_pred = evaluate_ensemble_multiple_random_forest(X_tr, Y_tr, X_ts, Y_ts, T_ts)
    Y_ts_pred_smooth = do_smooth_output(T_ts, Y_ts_pred)
    return Y_ts_pred_smooth
