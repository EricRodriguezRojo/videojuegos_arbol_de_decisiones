from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DEFAULT_DATASET_PATH = Path("data/videojuegos.csv")
DEFAULT_MODEL_PATH = Path("modelo.joblib")
TARGET = "Genre"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena modelos para clasificar el genero de videojuegos.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])
    return preprocessor, numeric_columns, categorical_columns


def main() -> None:
    args = parse_args()
    if not args.dataset_path.exists():
        raise FileNotFoundError(f"No existe el CSV: {args.dataset_path}")

    data = pd.read_csv(args.dataset_path).dropna(subset=[TARGET]).copy()
    X = data.drop(columns=[TARGET])
    y = data[TARGET].astype(str)
    preprocessor, numeric_columns, categorical_columns = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    decision_tree = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(max_depth=12, min_samples_leaf=2, random_state=RANDOM_STATE)),
    ])
    random_forest = Pipeline([
        ("preprocessor", build_preprocessor(X)[0]),
        ("classifier", RandomForestClassifier(
            n_estimators=140, max_depth=18, min_samples_leaf=1,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])

    print(f"Dataset: {args.dataset_path}")
    print(f"Muestras: {len(data)} | Variables: {len(X.columns)}")
    print("Entrenando arbol de decision...")
    decision_tree.fit(X_train, y_train)
    print("Entrenando random forest...")
    random_forest.fit(X_train, y_train)

    dt_predictions = decision_tree.predict(X_test)
    rf_predictions = random_forest.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print(f"Accuracy arbol: {dt_accuracy:.4f}")
    print(f"Accuracy random forest: {rf_accuracy:.4f}")
    print("\nReporte de clasificacion (Random Forest):")
    print(classification_report(y_test, rf_predictions, zero_division=0))

    metadata = {
        "dataset_type": "video-games-csv",
        "dataset_source": str(args.dataset_path.resolve()),
        "target": TARGET,
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "categorical_values": {
            column: sorted(X[column].dropna().astype(str).unique().tolist())
            for column in categorical_columns
        },
        "class_labels": sorted(y.unique().tolist()),
        "class_distribution": y.value_counts().sort_index().to_dict(),
        "metrics": {
            "decision_tree_accuracy": float(dt_accuracy),
            "random_forest_accuracy": float(rf_accuracy),
        },
    }
    artifact = {"decision_tree": decision_tree, "random_forest": random_forest, "metadata": metadata}
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_path, compress=3)
    print(f"\nModelo guardado en: {args.model_path.resolve()}")


if __name__ == "__main__":
    main()
