import shap, joblib
import pandas as pd, matplotlib.pyplot as plt
from src.utils.logger import logger

def explain_model():
    model = joblib.load("artifacts/xgboost_model.joblib")
    X_sample = pd.read_csv("data/processed/Test.csv").drop("Class", axis=1).sample(500, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("artifacts/shap_summary.png")
    logger.info("Saved SHAP summary plot to artifacts/shap_summary.png")

if __name__ == "__main__":
    explain_model()
