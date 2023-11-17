import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_data(train_path, test_path):
    df_train = pd.read_csv(train_path)  # 507, 148 -> first column is labels
    df_test = pd.read_csv(test_path)  # 168, 148

    x_train = df_train.drop(columns=['class']).values
    y_train = df_train['class'].values

    x_test = df_test.drop(columns=['class']).values
    y_test = df_test['class'].values

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    # Encode the categorical labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    return x_train, y_train_encoded, x_test, y_test_encoded
