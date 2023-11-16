"""
In this script I will compare different machine learning models to see
the performance of dimension reduction method mentioned in this study.
"""

# Imports
from xgboost import XGBClassifier
from knn.knn import knn
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import accuracy_score
import pandas as pd

from feature_selector import feature_selector
from knn.knn import knn
from optimizers.pso import pso
from problem import problem
from problem_terminate import problem_terminate
from read_data import read_data

# Variables
models = {
    'xgb': XGBClassifier(random_state=42, objective='accuracy'),
    'knn': knn,
    'lgbm': LGBMClassifier(random_state=42, objective='accuracy'),
    'rf': RandomForestClassifier(random_state=42),
    'hgb': HistGradientBoostingClassifier(random_state=42),
}


def get_model_accuracy(models, train, test):
    X_train, y_train = train
    X_test, y_test = test

    scores = {}

    for label, model in models.items():
        if not label == 'knn':
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            scores[label] = score
        else:
            error_rate = model(X_train, y_train, X_test, y_test, 5)
            scores[label] = 1-error_rate

    return scores


def main():
    train_path = "data\\urban_land_cover\\train.csv"
    test_path = "data\\urban_land_cover\\test.csv"
    X_train, y_train, X_test, y_test = read_data(train_path, test_path)

    # Get scores before dimension reduction
    scores = get_model_accuracy(models, (X_train, y_train), (X_test, y_test))

    g_best, history = pso(5, problem, problem_terminate, X_train, y_train, X_test, y_test)

    # Dimension reduction
    new_train, new_test = feature_selector(X_train, X_test, g_best['weights'][0])
    print(new_train.shape, new_test.shape)

    # Get scores after dimension reduction
    scores_after = get_model_accuracy(models, (new_train, y_train),
                                      (new_test, y_test))

    scores = pd.DataFrame(scores)
    scores_after = pd.DataFrame(scores_after)

    with pd.ExcelWriter('results.xlsx', engine='xlsxwriter') as writer:
        scores.to_excel(writer, sheet_name='Scores', index=False)
        scores_after.to_excel(writer, sheet_name='Scores_after', index=False)



if __name__ == '__main__':
    main()
