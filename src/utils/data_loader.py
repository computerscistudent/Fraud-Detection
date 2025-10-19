import pandas as pd
import os

def load_train_test_data():
    train_path = os.path.join("data","processed","train.csv")
    test_path = os.path.join("data","processed","test.csv")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    x_train = train_data.drop(columns=["Class"], axis=1)
    y_train = train_data["Class"]

    x_test = test_data.drop("Class", axis=1)
    y_test = test_data["Class"]

    return x_train, y_train, x_test, y_test