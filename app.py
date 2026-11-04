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
    st.title("Clasificador de Genero de Videojuegos")
    st.caption("Introduce seis datos sencillos y el modelo estimara su genero.")

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
    st.sidebar.metric("Macro F1 MLP adaptada", f"{metrics.get('mlp_adapted_macro_f1', 0):.3f}")
    model_mode = st.sidebar.selectbox(
        "Modelo",
        [
            "MLP base",
            "MLP adaptada",
            "Comparar MLP",
            "Arbol de Decision",
            "Random Forest",
            "Comparar todos",
        ],
    )

    with st.form("video_game_form"):
        values: dict[str, object] = {}
        columns = st.columns(2)
        input_labels = {
            "Name": "Nombre del videojuego",
            "Global_Sales": "Ventas globales (millones)",
            "Average_Score": "Media de puntuacion (0-100)",
            "Year_of_Release": "Ano de lanzamiento",
            "Publisher": "Publisher",
            "Developer": "Developer",
            "Rating": "Clasificacion por edades",
        }
        for index, column in enumerate(feature_columns):
            with columns[index % 2]:
                label = input_labels.get(column, column.replace("_", " "))
                if column == "Name":
                    values[column] = st.text_input(label, value="Nuevo videojuego")
                elif column == "Global_Sales":
                    values[column] = st.number_input(label, min_value=0.0, value=0.5, step=0.1)
                elif column == "Average_Score":
                    values[column] = st.number_input(
                        label, min_value=0.0, max_value=100.0, value=70.0, step=0.5
                    )
                elif column == "Year_of_Release":
                    values[column] = st.number_input(
                        label, min_value=1980, max_value=2030, value=2010, step=1
                    )
                else:
                    options = categorical_values.get(column, ["N/A"])
                    values[column] = st.selectbox(label, options)
        submitted = st.form_submit_button("Clasificar genero")

    if submitted:
        model_input = pd.DataFrame([values], columns=feature_columns)
        models = []
        if model_mode in ("MLP base", "Comparar MLP", "Comparar todos"):
            model = artifact["mlp_base"]
            probabilities = model.predict_proba(model_input)[0]
            index = probabilities.argmax()
            models.append(("MLP base", model.classes_[index], probabilities[index]))
        if model_mode in ("MLP adaptada", "Comparar MLP", "Comparar todos"):
            model = artifact["mlp_adapted"]
            probabilities = model.predict_proba(model_input)[0]
            index = probabilities.argmax()
            models.append(("MLP adaptada", model.classes_[index], probabilities[index]))
        if model_mode in ("Arbol de Decision", "Comparar todos"):
            model = artifact["decision_tree"]
            probabilities = model.predict_proba(model_input)[0]
            index = probabilities.argmax()
            models.append(("Arbol de Decision", model.classes_[index], probabilities[index]))
        if model_mode in ("Random Forest", "Comparar todos"):
            model = artifact["random_forest"]
            probabilities = model.predict_proba(model_input)[0]
            index = probabilities.argmax()
            models.append(("Random Forest", model.classes_[index], probabilities[index]))

        st.subheader("Resultado")
        for model_name, prediction, confidence in models:
            st.success(f"{model_name}: {prediction} ({confidence:.2%} de confianza)")


if __name__ == "__main__":
    main()
