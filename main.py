import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from knn.knn import knn
from optimizers.pso import pso
from problem import problem
from problem_terminate import problem_terminate
from feature_selector import feature_selector

# Read data
train_path = "data\\urban_land_cover\\train.csv"
test_path = "data\\urban_land_cover\\test.csv"
df_train = pd.read_csv(train_path)  # 507, 148 -> first column is labels
df_test = pd.read_csv(test_path)  # 168, 148

X_train = df_train.drop(columns=['class']).values
y_train = df_train['class'].values

X_test = df_test.drop(columns=['class']).values
y_test = df_test['class'].values

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Encode the categorical labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

error_rate1 = knn(X_train, y_train_encoded, X_test, y_test_encoded, 5)

print(f"Miss classification Error: %.2f" % error_rate1)

error_rate2 = knn(X_train, y_train_encoded, X_test, y_test_encoded, 5, np.ones((1, 147)))

print(f"Miss classification Error: %.2f" % error_rate2)

g_best, history = pso(5, problem, problem_terminate, X_train, y_train_encoded, X_test, y_test_encoded)

# Best mis classification error with weights
print(f"Best Error: %.2f" % g_best["fitness"])

# Dimension reduction
print(g_best['weights'][0])
new_train,new_test = feature_selector(X_train, X_test, g_best['weights'][0])

print(new_train.shape, new_test.shape)