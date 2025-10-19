import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import logger
from src.utils.exception import CustomException

RAW_PATH = os.path.join("data","raw","creditcard.csv")
PROCESS_DIR = os.path.join("data", "processed")
os.makedirs(PROCESS_DIR,exist_ok=True)

class DataIngestion:
    def __init__(self, raw_data_path="data/raw/creditcard.csv", processed_dir="data/processed"):
        self.raw_data_path = raw_data_path
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_data(self):
        try:
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Data loaded successfully with shape {df.shape}")
            return df
        except Exception as e:
            raise CustomException(e)
        
    def split_data(self,df):
        try:
            train , test = train_test_split(df,test_size=0.3,random_state=42,stratify=df["Class"])
            logger.info(
                f"✅ Data split successfully → Train: {train.shape}, Test: {test.shape}"
            )
            return train,test
        except Exception as e:
            raise CustomException(e) 

    def save_data(self,train_data,test_data):
        try:
            train_path = os.path.join(self.processed_dir,"train.csv")
            test_path = os.path.join(self.processed_dir,"test.csv")
            train_data.to_csv(train_path,index=False)
            test_data.to_csv(test_path,index=False)
            logger.info(f"✅ Train/Test data saved at {self.processed_dir}")
            print(f"Wrote processed train/test to {self.processed_dir}")
        except Exception as e:
            raise CustomException(e)       

if __name__ == "__main__":
    ingestion = DataIngestion()
    df = ingestion.load_data()
    train, test = ingestion.split_data(df)
    ingestion.save_data(train, test)