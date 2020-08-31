from sklearn.preprocessing import MinMaxScaler


def transfer_Scaler(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_mn = scaler.fit_transform(X_train)
    X_test_mn = scaler.transform(X_test)

    return X_train_mn, X_test_mn