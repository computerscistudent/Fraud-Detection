import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from src.utils.logger import logger
from src.utils.exception import CustomException

class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)

    def transform(self, X_train, y_train, X_test):
        try:
            X_res, y_res = self.smote.fit_resample(X_train, y_train)  # type: ignore
            logger.info(f"SMOTE applied - train shape after resampling: {X_res.shape}")

            X_res_scaled = self.scaler.fit_transform(X_res)  # type: ignore
            X_test_scaled = self.scaler.transform(X_test)

            X_res_scaled = pd.DataFrame(X_res_scaled, columns=X_train.columns)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

            return X_res_scaled, y_res, X_test_scaled

        except Exception as e:
            raise CustomException(e)