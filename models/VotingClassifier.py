from sklearn.neighbors import KNeighborsClassifier

from utils.util_import import *



def evaluate_voting(X_tr, Y_tr, X_ts, Y_ts, T_ts, n_estimators=100):

    w0 = len([v for v in Y_tr if v == 0])/len(list(Y_tr))
    w1 = len([v for v in Y_tr if v == 1])/len(list(Y_tr))

    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf2a = RandomForestClassifier(random_state=1, n_estimators=n_estimators)
    clf2b = RandomForestClassifier(random_state=1, n_estimators=n_estimators, class_weight={0:w1, 1:w0})
    clf2c = RandomForestClassifier(random_state=1, n_estimators=n_estimators, class_weight={0:w1, 1:w0})
    clf3 = GaussianNB()
    clf4 = KNeighborsClassifier()

    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('rf1', clf2a), ('rf2', clf2b), ('rf3', clf2c), ('gnb', clf3), ('knn', clf4)], voting='hard')

    eclf = eclf.fit(X_tr, Y_tr)

    Y_ts_pred = np.array(eclf.predict(X_ts))
    return Y_ts_pred

