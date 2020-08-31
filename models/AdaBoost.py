from utils.file_utils import load_model_from_file
from utils.smooth_output import do_smooth_output
from utils.util_import import *
from sklearn.svm import SVC

def evaluate_adaboost(X_tr, Y_tr, X_ts):

    # w0 = len([v for v in Y_tr if v == 0])/len(list(Y_tr))
    # w1 = len([v for v in Y_tr if v == 1])/len(list(Y_tr))

    # svc=SVC(probability=True, kernel='rbf')
    svc=SVC(probability=True, kernel='linear')
    # clf_AB = AdaBoostClassifier(n_estimators=100)
    clf_AB =AdaBoostClassifier(n_estimators=100, base_estimator=svc)
    clf_AB.fit(X_tr, Y_tr)

    Y_ts_pred = np.array(clf_AB.predict(X_ts))
    return Y_ts_pred


def get_trained_adaboost(X_tr, Y_tr):

    # svc=SVC(probability=True, kernel='rbf')
    # svc=SVC(probability=True, kernel='linear')
    clf_AB = AdaBoostClassifier(n_estimators=100)
    # clf_AB =AdaBoostClassifier(n_estimators=100, base_estimator=svc)
    clf_AB.fit(X_tr, Y_tr)

    return clf_AB


def run_adaboost(X_test):

    clf_AB = load_model_from_file(filename = 'trained_model_files/brushingAB.model')

    Y_pred = np.array(clf_AB.predict(X_test))
    return Y_pred

