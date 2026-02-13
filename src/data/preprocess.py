import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import yaml

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def clean_data(self, df):
        """Limpeza básica dos dados"""
        logger.info("Iniciando limpeza dos dados...")

        # Remover customerID (não é uma feature)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)

        # Converter TotalCharges para numérico, forçando erros a se tornarem NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # Preencher valores nulos em TotalCharges com a mediana (de forma segura)
        median_total_charges = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median_total_charges)
        logger.info(f"Valores nulos em TotalCharges preenchidos com a mediana: {median_total_charges}")

        # Converter Churn para binário
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

        # Verificar se ainda existem NaNs em outras colunas
        cols_com_nan = df.columns[df.isna().any()].tolist()
        if cols_com_nan:
            logger.warning(
                f"Atenção: As seguintes colunas ainda contêm NaN após limpeza inicial: {cols_com_nan}. Eles serão tratados na imputação.")
        else:
            logger.info("Nenhum valor NaN encontrado após limpeza inicial.")

        logger.info(f"Limpeza concluída. Shape: {df.shape}")
        return df

    def split_data(self, df):
        """Divide dados em treino e teste"""
        X = df.drop('Churn', axis=1)
        y = df['Churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )

        logger.info(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
        return X_train, X_test, y_train, y_test