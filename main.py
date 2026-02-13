import logging
import sys
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar módulos
from src.data.make_dataset import DataLoader
from src.data.preprocess import DataPreprocessor
from src.features.build_features import FeatureEngineer
from src.models.train_model import ModelTrainer


def main():
    """Função principal do projeto"""

    print("\n" + "=" * 60)
    print(" PROJETO DE PREDIÇÃO DE CHURN")
    print("=" * 60 + "\n")

    try:
        # 1. Carregar dados
        print("1. Carregando dados...")
        loader = DataLoader()
        df = loader.load_data()
        print(f"   ✓ Dados carregados: {df.shape}")

        # 2. Pré-processar
        print("\n2. Pré-processando dados...")
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data(df_clean)
        print(f"   ✓ Treino: {X_train.shape}, Teste: {X_test.shape}")

        # 3. Engenharia de features com imputação
        print("\n3. Engenharia de features...")
        feature_eng = FeatureEngineer()
        X_train_proc, X_test_proc = feature_eng.fit_transform(X_train, X_test)
        feature_eng.save_preprocessor()
        print(f"   ✓ Features processadas: {X_train_proc.shape[1]}")

        # 4. Treinar modelos
        print("\n4. Treinando modelos...")
        trainer = ModelTrainer()
        results = trainer.train_all(X_train_proc, y_train, X_test_proc, y_test)

        # 5. Resultados
        print("\n" + "=" * 60)
        print(" RESULTADOS")
        print("=" * 60)

        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  - Acurácia: {metrics['accuracy']:.4f}")
            print(f"  - Precisão: {metrics['precision']:.4f}")
            print(f"  - Recall: {metrics['recall']:.4f}")
            print(f"  - F1-Score: {metrics['f1']:.4f}")
            print(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")

        print(f"\n✅ Melhor modelo: {trainer.best_model}")
        trainer.save_model()

        print("\n" + "=" * 60)
        print(" PROJETO CONCLUÍDO COM SUCESSO!")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Erro durante a execução: {e}")
        raise


if __name__ == "__main__":
    main()