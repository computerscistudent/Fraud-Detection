import os
import pandas as pd

def test_data_files_exist():
    assert os.path.exists("data/processed/train.csv")
    assert os.path.exists("data/processed/test.csv")

def test_class_column_present():
    df = pd.read_csv("data/processed/train.csv")
    assert "Class" in df.columns