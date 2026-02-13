from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

logger = logging.getLogger(__name__)


def tune_random_forest(X_train, y_train):
    """Otimiza hiperparâmetros do Random Forest"""

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', None]
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    logger.info(f"Melhores parâmetros: {grid_search.best_params_}")
    logger.info(f"Melhor F1-score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_4