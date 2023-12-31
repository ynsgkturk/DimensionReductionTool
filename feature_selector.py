
def feature_selector(train, test, weights, threshold=0.1):
    """
    Selects features that are important to the models

    :param train: Train dataset
    :param test: Test dataset
    :param weights: Feature weights
    :param threshold: Threshold for selecting features. It is 0.1 by default
    :return: Train and Test dataset with selected features.
    """

    indices = weights >= threshold

    new_train = train[:, indices]
    new_test = test[:, indices]

    return new_train, new_test
