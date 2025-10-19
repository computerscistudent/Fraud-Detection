💳 Credit Card Fraud Detection System
🧠 Detect fraudulent transactions using Machine Learning and FastAPI

📘 Overview

This project predicts the likelihood of a credit card transaction being fraudulent using advanced machine learning models.
It includes EDA, model training, explainability (SHAP), and a user-friendly FastAPI web app for CSV-based predictions.

🚀 Features

✅ Data Preprocessing & EDA
✅ Baseline Model (Logistic Regression)
✅ Advanced Models — Random Forest & XGBoost
✅ SMOTE for handling class imbalance
✅ Explainability using SHAP
✅ Web Interface for uploading CSV files
✅ API endpoints for developers
✅ Deployment-ready structure

🗂️ Project Structure

Fraud_detection/
│
├── data/
│   ├── raw/creditcard.csv
│   ├── processed/Train.csv
│   └── processed/Test.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── artifacts/
│   ├── xgboost_model.joblib
│   ├── randomforest_model.joblib
│   └── shap_summary.png
│
├── src/
│   ├── api/
│   │   ├── templates/
│   │   ├── static/
│   │   └── main.py
│   ├── models/
│   ├── utils/
│
├── requirements.txt
├── README.md
└── .gitignore


📊 Model Performance
Model	               ROC AUC	 Precision Mean

Logistic Regression	   0.9679	 0.0159
🎋 Random Forest	  0.9773	0.0117
⚡ XGBoost	         0.9783	   0.0117

🔹 XGBoost achieved the best ROC AUC score, showing strong discrimination between fraud and legitimate transactions.

🧩 Tech Stack

Python 3.10+

FastAPI (backend framework)

Scikit-learn, XGBoost, imblearn, SHAP

Pandas, NumPy, Matplotlib

Jinja2 Templates for frontend

Render / Hugging Face Spaces for deployment


⚙️ How to Run Locally

# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/Fraud_detection.git
cd Fraud_detection

# 2️⃣ Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run FastAPI app
uvicorn src.api.main:app --reload

Then open your browser at 👉 http://127.0.0.1:8000/


🧮 API Endpoints
Route	             Method	         Description
/	                 GET	         Web interface for CSV upload
/predict_file	     POST	         Predict fraud probability from uploaded CSV
/docs	             GET	         Swagger UI for developers


🧠 Explainability (SHAP)

Feature importance visualization generated via SHAP for interpretability.

Output saved as → artifacts/shap_summary.png


🌐 Deployment

The project is deployed on Render for public access:
👉 Fraud Detection App (Live)


🧾 License

This project is released under the MIT License.


👨‍💻 Author

Abhimanyu Singh
💼 Machine Learning & Software Engineer
📧 abhimanyu@example.com

🌐 LinkedIn
 | GitHub


💬 Acknowledgments

Dataset: Credit Card Fraud Detection (Kaggle)