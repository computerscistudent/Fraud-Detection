import joblib
import pandas as pd

def test_model_predicts():
    model = joblib.load("artifacts/baseline_model.joblib")
    df = pd.read_csv("data/processed/test.csv").head(10)
    X = df.drop(columns=["Class"])
    preds = model.predict_proba(X)
    assert preds.shape[0] == 10