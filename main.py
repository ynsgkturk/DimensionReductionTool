from feature_selector import feature_selector
from knn.knn import knn
from optimizers.pso import pso
from problem import problem
from problem_terminate import problem_terminate
from read_data import read_data

# Read data
train_path = "data\\urban_land_cover\\train.csv"
test_path = "data\\urban_land_cover\\test.csv"
X_train, y_train, X_test, y_test = read_data(train_path, test_path)


error_rate1 = knn(X_train, y_train, X_test, y_test, 5)
print(f"Miss classification Error: %.2f" % error_rate1)

g_best, history = pso(5, problem, problem_terminate, X_train, y_train, X_test, y_test)
print(f"Best Error Found: %.2f" % g_best["fitness"])

# Dimension reduction
new_train, new_test = feature_selector(X_train, X_test, g_best['weights'][0])
print(new_train.shape, new_test.shape)

# Get the error after dropping minor features
error_rate_new = knn(new_train, y_train, new_test, y_test, 5)
print(f"Miss classification error after dropping minor features: %.2f" % error_rate_new)

g_best_new, history_new = pso(5, problem, problem_terminate, new_train, y_train, new_test, y_test)
print(f"Miss classification error after dropping minor features: %.2f" % g_best_new["fitness"])
