import os
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, classification_report, confusion_matrix,average_precision_score

from src.utils.logger import logger
from src.utils.exception import CustomException

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

def precision_at_k(y_true,probs,k=0.1):
    n = len(probs)
    top_n = max(1,int(n*k))
    idx = np.argsort(probs)[::-1][:top_n]
    return y_true.iloc[idx].sum() / top_n

def main():
    try:
        logger.info("Starting evaluation")
        df_test = pd.read_csv("data/processed/Test.csv") 
        X_test = df_test[[c for c in df_test.columns if c != "Class"]]
        y_test = df_test["Class"]

        model = joblib.load(ARTIFACTS_DIR / "baseline_model.joblib")
        probs = model.predict_proba(X_test)[:,1]
        preds = (probs >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, probs)
        logger.info(f"Test ROC AUC: {roc_auc:.4f}")
        print("ROC AUC:", roc_auc)

        print(classification_report(y_test, preds))
        print("Confusion matrix:\n", confusion_matrix(y_test, preds))

        p_at_10 = precision_at_k(y_test.reset_index(drop=True), probs, k=0.1)
        print("Precision@10%:", p_at_10)

        fpr,tpr,_ = roc_curve(y_test,probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
        plt.plot([0,1],[0,1], '--', color='gray')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC curve")
        plt.legend()
        plt.savefig(ARTIFACTS_DIR / "roc_curve.png")

        precision, recall, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        plt.figure()
        plt.plot(recall, precision, label=f"AP = {ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.savefig(ARTIFACTS_DIR / "pr_curve.png")


        report_str = classification_report(y_test, preds)
        metrics = {
            "roc_auc":roc_auc,
            "p_at_10":p_at_10,
            "classification_report": report_str
        }
        with open(ARTIFACTS_DIR / "test_metrics.json", "w") as f:
            json.dump(metrics,f,indent=2)
        logger.info("Evaluation complete. Artifacts saved.")
    except Exception as e:
        raise CustomException(e)
    
if __name__ == "__main__":
    main()    