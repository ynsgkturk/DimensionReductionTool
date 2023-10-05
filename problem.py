from knn.knn import knn


def problem(weights, X_train, y_train, X_test, y_test, k):
    """
    Calls Knn function to evaluate weights found by optimizer.

    :param weights: Feature weights found by optimizer.
    :param X_train: Train dataset
    :param y_train: Target values for train.
    :param X_test: Test dataset.
    :param y_test: Target values for test.
    :param k: Neighbor number.

    :return: Miss classification error calculated by knn function.
    """
    return knn(X_train, y_train, X_test, y_test, k, weights)