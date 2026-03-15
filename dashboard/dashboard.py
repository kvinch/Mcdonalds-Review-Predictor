import os
import joblib
import pandas as pd
import streamlit as st

def main():
    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "model", "rf_model.pkl")

    st.set_page_config(
        page_title="McDonald's Review Predictor",
        layout="wide",
    )

    @st.cache_resource
    def load_model():
        payload = joblib.load(MODEL_PATH)
        return payload["model"], payload["class_names"], payload["feature_names"]

    rf, class_names, feature_names = load_model()
    st.title("McDonald's Review Predictor")
    st.caption("Modelo Random Forest | Clasifica una reseña como **good**, **neutral** o **bad**.")
    st.divider()

    st.subheader("Rendimiento del modelo")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy TEST",      "72.6%")
    c2.metric("Cross-Val (5-fold)", "72.3%")
    c3.metric("Estimadores RF",     "300")

    st.divider()

    st.subheader("Insights clave")

    with st.expander("Ver insights del modelo", expanded=False):
        st.markdown("""
    - **Sentimiento** (`sentiment_polarity`, `sentiment_compound`) son las features más importantes. El tono del texto predice bien la satisfacción del cliente.
    - **Longitud de reseña** (`review_length`, `review_word_count`): los clientes insatisfechos suelen a escribir reseñas más largas.
    - **Recencia** (`review_time_since_days`): variable contínua que captura el contexto temporal de la reseña.
    - **Geografía** (`city_enc`, `postal_code_enc`): baja importancia Gini — omitidas del formulario de predicción.
    - **Desbalance de clases**: bad 41.7% · neutral 29.9% · good 28.5%. El modelo usa `class_weight='balanced'`.
    """)

    st.divider()
    st.subheader("Datasets")
    options = ["Raw","Limpio","Preprocesado"]
    choice = st.selectbox("Escoga el Dataset", options)

    if choice == "Raw":
        st.subheader("Raw")
        st.dataframe(pd.read_csv(os.path.join(BASE_DIR, "data","McDonalds_RAW.csv"),encoding ="latin1"))
    elif choice == "Limpio":
        st.subheader("Limpio")
        st.dataframe(pd.read_csv(os.path.join(BASE_DIR, "data", "McDonalds_Clean.csv"),encoding ="latin1"))
    elif choice == "Preprocesado":
        st.subheader("Preprocesado")
        st.dataframe(pd.read_csv(os.path.join(BASE_DIR, "data","processed", "McDonalds_Features.csv"),encoding ="latin1"))
    
    st.divider()

    st.subheader("Prediccion interactiva")
    st.caption("Introduce los valores de la resena y presione **Predecir**.")
    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Sentimiento del texto**")
            sentiment_compound = st.slider("Compound VADER (-1 a +1)",      -1.0, 1.0, 0.0, 0.01)
            sentiment_polarity = st.slider("Polarity TextBlob (-1 a +1)",   -1.0, 1.0, 0.0, 0.01)
            sentiment_subjectivity= st.slider("Subjectivity TextBlob (0 a 1)",  0.0, 1.0, 0.5, 0.01)
            sentiment_label= st.selectbox(
                "Clasificacion VADER",
                options=["positive", "neutral", "negative"],
                index=1,
            )

        with col_b:
            st.markdown("**Caracteristicas de la resena**")
            review_length          = st.number_input("Longitud (caracteres)", min_value=1,  max_value=5000, value=150, step=10)
            review_word_count      = st.number_input("Palabras",              min_value=1,  max_value=1000, value=30,  step=5)
            review_time_since_days = st.number_input("Dias desde la resena",  min_value=1,  max_value=3650, value=180, step=30)

            st.markdown("**Local**")
            rating_count     = st.number_input("Ratings del local", min_value=1, max_value=5000, value=500, step=50)
            location_cluster = st.selectbox("Cluster de ubicacion (0-9)", options=list(range(10)), index=0)

        submitted = st.form_submit_button("Predecir")

    if submitted:
        senti_negative = 1 if sentiment_label == "negative" else 0
        senti_neutral  = 1 if sentiment_label == "neutral"  else 0
        senti_positive = 1 if sentiment_label == "positive" else 0

        input_dict = {
            "rating_count":           rating_count,
            "review_time_since_days": review_time_since_days,
            "review_length":          review_length,
            "location_cluster":       location_cluster,
            "sentiment_compound":     sentiment_compound,
            "sentiment_polarity":     sentiment_polarity,
            "sentiment_subjectivity": sentiment_subjectivity,
            "review_word_count":      review_word_count,
            "city_enc":               0,
            "postal_code_enc":        0,
            "senti_negative":         senti_negative,
            "senti_neutral":          senti_neutral,
            "senti_positive":         senti_positive,
        }

        X_input    = pd.DataFrame([input_dict])[feature_names]
        pred_idx   = rf.predict(X_input)[0]
        pred_proba = rf.predict_proba(X_input)[0]
        pred_label = class_names[pred_idx]

        st.divider()
        st.subheader("Resultado")

        colors = {"bad": "red", "neutral": "orange", "good": "green"}
        st.markdown(
            f"<h2 style='color:{colors[pred_label]};text-align:center'>{pred_label.upper()}</h2>",
            unsafe_allow_html=True,
        )

        st.markdown("**Probabilidades por clase:**")
        for cls, prob in zip(class_names, pred_proba):
            st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

if __name__ == "__main__":
    main()