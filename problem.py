from knn.knn import knn


def problem(weights, X_train, y_train, X_test, y_test, k):
    return knn(X_train, y_train, X_test, y_test, k, weights)