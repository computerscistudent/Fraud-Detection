from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd, joblib, io
from src.utils.logger import logger
from fastapi.staticfiles import StaticFiles
import numpy as np

app = FastAPI(title="Fraud Detection API")
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")
templates = Jinja2Templates(directory="src/api/templates")

MODEL_PATH = "artifacts/xgboost_model.joblib"
model = joblib.load(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
def home(request:Request):
    return templates.TemplateResponse("home.html",{"request": request})

@app.get("/index", response_class=HTMLResponse)
def index(request:Request):
    return templates.TemplateResponse("index.html",{"request": request})

@app.post("/submit", response_class=HTMLResponse)
async def predict_file(request:Request, file:UploadFile=File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        preds = model.predict_proba(df)[:,1]
        df["fraud_probablity"] = np.round(preds*100,2)
        df["label"] = df["fraud_probablity"].apply(lambda x : "⚠️ Fraud" if x>50 else "✅ Legit")
        fraud_count = (df["label"]=="⚠️ Fraud").sum()
        total = len(df)
        logger.info(f"Predicted {fraud_count} frauds out of {total} records")
        table_html = df.head(20).to_html(classes="styled-table", index=False)
        return templates.TemplateResponse("result.html",{"request": request,
                                                        "table": table_html,
                                                        "fraud_count": fraud_count,
                                                        "total": total,})
    except Exception as e:
        logger.error(e)
        return templates.TemplateResponse("result.html", {"request": request, "error": str(e)})
    
@app.get("/docs_redirect", response_class=HTMLResponse)
def redirect_docs(request:Request):
    return templates.TemplateResponse("docs_redirect.html",{"request": request})
