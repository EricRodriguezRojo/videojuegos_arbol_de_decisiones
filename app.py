from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("modelo.joblib")


@st.cache_resource
def load_artifact() -> dict:
    return joblib.load(MODEL_PATH)


def main() -> None:
    st.set_page_config(page_title="Clasificador de generos", page_icon="🎮", layout="wide")
    st.title("🎮 Clasificador de genero de videojuegos")
    st.caption("Introduce las caracteristicas de un videojuego y el modelo estimara su genero.")

    if not MODEL_PATH.exists():
        st.error("No existe modelo.joblib. Ejecuta primero: python train.py")
        st.stop()

    artifact = load_artifact()
    metadata = artifact["metadata"]
    feature_columns = metadata["feature_columns"]
    numeric_columns = set(metadata["numeric_columns"])
    categorical_values = metadata["categorical_values"]
    metrics = metadata.get("metrics", {})

    st.sidebar.metric("Accuracy arbol", f"{metrics.get('decision_tree_accuracy', 0):.3f}")
    st.sidebar.metric("Accuracy random forest", f"{metrics.get('random_forest_accuracy', 0):.3f}")
    model_mode = st.sidebar.selectbox("Modelo", ["Arbol de Decision", "Random Forest", "Comparar ambos"])

    with st.form("video_game_form"):
        values: dict[str, object] = {}
        columns = st.columns(3)
        for index, column in enumerate(feature_columns):
            with columns[index % 3]:
                label = column.replace("_", " ")
                if column in numeric_columns:
                    values[column] = st.number_input(label, value=0.0, format="%.2f")
                elif column == "Name":
                    values[column] = st.text_input(label, value="Nuevo videojuego")
                else:
                    options = categorical_values.get(column, ["N/A"])
                    values[column] = st.selectbox(label, options)
        submitted = st.form_submit_button("Clasificar genero")

    if submitted:
        model_input = pd.DataFrame([values], columns=feature_columns)
        models = []
        if model_mode in ("Arbol de Decision", "Comparar ambos"):
            model = artifact["decision_tree"]
            probabilities = model.predict_proba(model_input)[0]
            index = probabilities.argmax()
            models.append(("Arbol de Decision", model.classes_[index], probabilities[index]))
        if model_mode in ("Random Forest", "Comparar ambos"):
            model = artifact["random_forest"]
            probabilities = model.predict_proba(model_input)[0]
            index = probabilities.argmax()
            models.append(("Random Forest", model.classes_[index], probabilities[index]))

        st.subheader("Resultado")
        for model_name, prediction, confidence in models:
            st.success(f"{model_name}: {prediction} ({confidence:.2%} de confianza)")


if __name__ == "__main__":
    main()
