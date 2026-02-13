from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI(title="Churn Prediction API", description="API para predição de churn")

# Carregar modelo e pré-processador na inicialização
model_path = Path("models/LogisticRegression.joblib")
preprocessor_path = Path("models/preprocessor.joblib")

if model_path.exists() and preprocessor_path.exists():
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print("✅ Modelo e pré-processador carregados!")
else:
    print("❌ Modelo ou pré-processador não encontrado. Execute main.py primeiro.")
    model = None
    preprocessor = None


# Definir modelo de dados
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    churn: str
    probability: float
    risk_level: str


@app.get("/")
def read_root():
    return {"message": "Churn Prediction API", "status": "running"}


@app.get("/health")
def health_check():
    if model and preprocessor:
        return {"status": "healthy", "model": "loaded"}
    return {"status": "unhealthy", "model": "not loaded"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

    try:
        # Converter para DataFrame
        df = pd.DataFrame([customer.dict()])

        # Pré-processar
        X_processed = preprocessor.transform(df)

        # Predizer
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0][1]

        # Determinar nível de risco
        if probability > 0.6:
            risk = "Alto"
        elif probability > 0.3:
            risk = "Médio"
        else:
            risk = "Baixo"

        return PredictionResponse(
            churn="Sim" if prediction == 1 else "Não",
            probability=round(float(probability), 4),
            risk_level=risk
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Para executar: uvicorn api:app --reload