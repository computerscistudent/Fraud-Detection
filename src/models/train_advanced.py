import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.models.feature_engineering import FeatureEngineering
from src.utils.data_loader import load_train_test_data  # small helper you can write

def train_models():
    x_train, y_train, x_test, y_test = load_train_test_data()

    fe = FeatureEngineering()
    X_res, y_res, X_test_scaled =  fe.transform(x_train, y_train, x_test)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='auc')
    }
    results = {}
    for name,model in models.items():
        logger.info(f"Training {name} ...")
        model.fit(X_res,y_res)
        preds = model.predict_proba(X_test_scaled)[:,1]
        auc = roc_auc_score(y_test, preds)
        precision, recall, _ = precision_recall_curve(y_test, preds)
        results[name] = {"roc_auc": auc, "precision_mean": np.mean(precision)}
        joblib.dump(model, f"artifacts/{name.lower()}_model.joblib")
        logger.info(f"{name} – ROC AUC: {auc:.4f}")
    return results    

if __name__ == "__main__":
    results = train_models()
    print("\nModel Comparison Summary:")
    for name, metrics in results.items(): 
        print(f"{name}: ROC AUC = {metrics['roc_auc']:.4f}, Precision Mean = {metrics['precision_mean']:.4f}")    
