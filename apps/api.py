from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.modeling.predictor import ChurnPredictor, PredictionResult

predictor = ChurnPredictor()


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
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
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    churn: str
    probability: float
    risk_level: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        predictor.load_artifacts()
    except FileNotFoundError:
        # API segue no ar para facilitar health-check mesmo sem artefatos.
        pass
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="API para predição de churn",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Churn Prediction API", "status": "running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy" if predictor.is_ready else "unhealthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData) -> PredictionResponse:
    try:
        result: PredictionResult = predictor.predict_from_dict(customer.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erro ao prever churn: {exc}") from exc

    return PredictionResponse(
        churn=result.churn,
        probability=result.probability,
        risk_level=result.risk_level,
    )
