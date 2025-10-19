import os
import sys
import json
from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.utils.logger import logger
from src.utils.exception import CustomException

ARTIFACTS_DIR = Path("artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def get_feature_column(df):
    return [c for c in df.columns if c != "Class"]

def build_pipeline(numerical_to_scale):
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(),numerical_to_scale)
        ], remainder= "passthrough"
    )
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000))
        ]
    )
    return pipe

def main():
    try:
        logger.info("Starting training pipeline")
        df = pd.read_csv("data/processed/train.csv")
        features = get_feature_column(df)
        X = df[features]
        y = df["Class"]

        scale_cols = [c for c in ["Amount", "Time"] if c in X.columns]
        pipe = build_pipeline(scale_cols)

        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        logger.info("Running cross-validation")
        scores = cross_val_score(pipe,X,y,cv=cv,scoring="roc_auc",n_jobs=-1)
        mean_scores = float(scores.mean())
        logger.info(f"CV ROC AUC scores: {scores}, mean: {mean_scores:.4f}")
        print("CV ROC AUC:", scores, mean_scores)

        pipe.fit(X,y)
        joblib.dump(pipe,ARTIFACTS_DIR/"baseline_model.joblib")
        logger.info(f"Saved baseline model to {ARTIFACTS_DIR/'baseline_model.joblib'}")

        metrics = {"cv_scores": scores.tolist(), "cv_mean_auc": mean_scores}
        with open(ARTIFACTS_DIR/"baseline_metrics.json" , "w") as f:
            json.dump(metrics,f,indent=2)
    except Exception as e:
        logger.exception("Training failed")
        raise CustomException(e)
    
if __name__ == "__main__":
    main()