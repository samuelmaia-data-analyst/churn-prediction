import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import yaml
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.preprocessor = None
        self.feature_names = None

    def create_preprocessor(self):
        """Cria pipeline de pré-processamento com imputação"""

        # Pipeline para features numéricas: Imputação + Escalonamento
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Pipeline para features categóricas: Imputação + One-Hot Encoding
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

        # Combinar
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.config['features']['numerical_features']),
                ('cat', categorical_transformer, self.config['features']['categorical_features'])
            ])

        return self.preprocessor

    def fit_transform(self, X_train, X_test):
        """Ajusta e transforma os dados"""
        if self.preprocessor is None:
            self.create_preprocessor()

        logger.info("Ajustando pré-processador...")
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)

        # Gerar nomes das features
        self._generate_feature_names(X_train)

        # Converter para DataFrame com nomes de colunas
        X_train_proc = pd.DataFrame(
            X_train_proc,
            columns=self.feature_names,
            index=X_train.index
        )
        X_test_proc = pd.DataFrame(
            X_test_proc,
            columns=self.feature_names,
            index=X_test.index
        )

        logger.info(f"Features processadas: {X_train_proc.shape[1]}")
        return X_train_proc, X_test_proc

    def _generate_feature_names(self, X_train):
        """Gera nomes para as features após transformação"""
        feature_names = []

        # Nomes das features numéricas (mantém os originais)
        feature_names.extend(self.config['features']['numerical_features'])

        # Nomes das features categóricas após one-hot
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = self.config['features']['categorical_features']

        for i, cat_feat in enumerate(cat_features):
            # Obter categorias únicas da coluna (excluindo a primeira por causa do drop='first')
            unique_vals = X_train[cat_feat].dropna().unique()
            # Para cada valor único (exceto o primeiro), criar um nome de feature
            for j, val in enumerate(sorted(unique_vals)[1:], 1):
                feature_names.append(f"{cat_feat}_{val}")

        self.feature_names = feature_names

    def save_preprocessor(self, path="models/preprocessor.joblib"):
        """Salva o pré-processador"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, path)
        logger.info(f"Pré-processador salvo em: {path}")

    def load_preprocessor(self, path="models/preprocessor.joblib"):
        """Carrega o pré-processador"""
        self.preprocessor = joblib.load(path)
        logger.info(f"Pré-processador carregado de: {path}")