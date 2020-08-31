
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils.smooth_output import do_smooth_output
from utils.util_import import *

def evaluate_DT(X_tr, Y_tr, X_ts, Y_ts, T_ts):

    clf = DecisionTreeClassifier()

    clf.fit(X_tr, Y_tr)

    Y_ts_pred = np.array(clf.predict(X_ts))
    return Y_ts_pred



