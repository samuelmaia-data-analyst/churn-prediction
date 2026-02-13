import pandas as pd
import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def load_data(self):
        """Carrega dados do arquivo CSV"""
        try:
            df = pd.read_csv(self.config['data']['raw_path'])
            logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
            return df
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            raise

    def save_processed(self, df):
        """Salva dados processados"""
        output_path = self.config['data']['processed_path']
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Dados salvos em: {output_path}")