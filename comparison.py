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
from scipy.io import loadmat

from feature_selector import feature_selector
from optimizers.pso import pso
from problem import problem
from problem_terminate import problem_terminate
from read_data import read_data

import warnings

# Variables
models = {
    'xgb': XGBClassifier(random_state=42, objective='accuracy'),
    'knn': knn,
    'lgbm': LGBMClassifier(random_state=42, objective='multiclass', verbose=-1),
    'rf': RandomForestClassifier(random_state=42),
    'hgb': HistGradientBoostingClassifier(random_state=42),
}


def get_model_accuracy(models, train, test):
    x_train, y_train = train
    x_test, y_test = test

    scores = {}

    for label, model in models.items():
        if not label == 'knn':
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            score = accuracy_score(y_test, y_pred)

            scores[label] = score
        else:
            error_rate = model(x_train, y_train, x_test, y_test, 5)
            scores[label] = 1 - error_rate

    return scores


def main():
    warnings.filterwarnings('ignore')

    train_path = "data\\urban_land_cover\\train.csv"
    test_path = "data\\urban_land_cover\\test.csv"
    x_train, y_train, x_test, y_test = read_data(train_path, test_path)

    # Get scores before dimension reduction
    scores = get_model_accuracy(models, (x_train, y_train), (x_test, y_test))

    # PSO
    g_best, history = pso(5, problem, problem_terminate, x_train, y_train, x_test, y_test)

    # Dimension reduction
    new_train, new_test = feature_selector(x_train, x_test, g_best['weights'][0])
    print(new_train.shape, new_test.shape)

    # Get scores after dimension reduction
    scores_pso = get_model_accuracy(models, (new_train, y_train),
                                    (new_test, y_test))

    # FDB-AOA
    fdb_aoa_weights = loadmat(r'weights\weights.mat')

    new_train, new_test = feature_selector(x_train, x_test, fdb_aoa_weights['bestWeights'][0])
    print(new_train.shape, new_test.shape)

    # Get scores after dimension reduction
    scores_aoa = get_model_accuracy(models, (new_train, y_train),
                                    (new_test, y_test))

    scores = pd.DataFrame({'model': scores.keys(),
                           'score': scores.values(),
                           'pso_score': scores_pso.values(),
                           'fdb_aoa_score': scores_aoa.values()})

    with pd.ExcelWriter('results.xlsx', engine='xlsxwriter') as writer:
        scores.to_excel(writer, sheet_name='Scores', index=False)


if __name__ == '__main__':
    main()
