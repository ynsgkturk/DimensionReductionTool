import numpy as np


def knn(X_train, y_train, X_test, y_test, k, weights=None):
    if weights is None:
        # Compute Euclidean distances between each test point and all training points
        distances = np.sqrt(np.sum((X_train - X_test[:, np.newaxis]) ** 2, axis=2))
    else:
        # Calculate the weighted Euclidean distances between each test point and all training points
        squared_diff = (X_train - X_test[:, np.newaxis]) ** 2
        weighted_squared_diff = squared_diff * weights
        distances = np.sqrt(np.sum(weighted_squared_diff, axis=2))

    # Find k nearest neighbors for each test point
    nearest_indices = np.argsort(distances, axis=1)[:, :k]

    # Get the labels of the k nearest neighbors
    nearest_labels = y_train[nearest_indices]

    # Predict the labels of the test points based on majority voting
    def majority_vote(labels):
        counts = np.bincount(labels)
        return np.argmax(counts)

    predicted_labels = [majority_vote(row) for row in nearest_labels]

    # Calculate error rate
    error_rate = np.sum(predicted_labels != y_test) / len(y_test)

    return error_rate
