from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def tune_random_forest(X_train, y_train):
    """Otimiza hiperparametros do RandomForest e retorna o melhor estimador."""

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", None],
    }

    model = RandomForestClassifier(random_state=42)
    search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    return search.best_estimator_
