import joblib
import pandas as pd
import numpy as np
from pathlib import Path


def predict_single_customer(customer_data):
    """
    Faz predição para um único cliente

    Args:
        customer_data (dict): Dicionário com dados do cliente

    Returns:
        dict: Resultado da predição
    """

    # Carregar modelo e pré-processador
    model_path = Path("models/LogisticRegression.joblib")
    preprocessor_path = Path("models/preprocessor.joblib")

    if not model_path.exists() or not preprocessor_path.exists():
        print("❌ Modelo ou pré-processador não encontrado. Execute main.py primeiro.")
        return

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Converter para DataFrame
    df = pd.DataFrame([customer_data])

    # Pré-processar
    X_processed = preprocessor.transform(df)

    # Predizer
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0]

    # Resultado
    result = {
        'churn': 'Sim' if prediction == 1 else 'Não',
        'probabilidade': probability[1],
        'risco': 'Alto' if probability[1] > 0.6 else 'Médio' if probability[1] > 0.3 else 'Baixo'
    }

    return result


# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo de um cliente
    cliente_teste = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 65.5,
        'TotalCharges': 786.0
    }

    resultado = predict_single_customer(cliente_teste)

    if resultado:
        print("\n" + "=" * 50)
        print(" RESULTADO DA PREDIÇÃO")
        print("=" * 50)
        print(f"Churn: {resultado['churn']}")
        print(f"Probabilidade: {resultado['probabilidade']:.2%}")
        print(f"Nível de Risco: {resultado['risco']}")