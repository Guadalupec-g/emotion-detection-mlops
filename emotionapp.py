import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Título
st.title("🧠 Detección de Emociones con IA")
st.markdown("Sube un archivo CSV con tus señales EEG o responde un formulario para estimar tu estado emocional: 😄, 😐 o 😠")

# Cargar modelo y scaler
@st.cache_resource
def load_model():
    model = joblib.load("model/random_forest_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_model()

# Opcion 1: Subir archivo CSV
st.header("📂 Opción 1: Subir archivo EEG")
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        if data.shape[1] != 2548:
            st.error("⚠️ El archivo debe tener exactamente 2548 columnas de características.")
        else:
            X_scaled = scaler.transform(data)
            prediction = model.predict(X_scaled)
            predicted_emotions = label_encoder.inverse_transform(prediction)
            st.success("Predicciones completadas ✅")
            st.write(pd.DataFrame({"Predicción de Emoción": predicted_emotions}))
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")

# Opcion 2: Test emocional interactivo
st.header("🧠 Opción 2: Test de Autoevaluación Emocional")
st.markdown("Responde estas preguntas sobre tu estado actual para predecir tu emoción")

form = st.form(key="emotion_form")

energy = form.slider("¿Cuánta energía mental sientes ahora?", 0, 10, 5)
stress = form.slider("¿Te sientes estresada o preocupada?", 0, 10, 5)
focus = form.slider("¿Qué tan enfocada estás en este momento?", 0, 10, 5)
rest = form.slider("¿Te sientes descansada y tranquila?", 0, 10, 5)
happiness = form.slider("¿Te sientes feliz o positiva?", 0, 10, 5)
irritability = form.slider("¿Te sientes irritable o frustrada?", 0, 10, 5)
calm = form.slider("¿Te sientes en calma interior?", 0, 10, 5)

submit = form.form_submit_button("Predecir Emoción")

if submit:
    # Algoritmo simple basado en puntuaciones
    score = happiness + calm + rest - stress - irritability
    if score >= 10:
        emotion = "POSITIVE"
        message = "😄 Estás en un estado emocional positivo. Ideal para meditar, crear y visualizar." \
                  " Prueba una afirmación como: *'Estoy en perfecta armonía y equilibrio mental.'*"
    elif score <= -5:
        emotion = "NEGATIVE"
        message = "😠 Parece que estás pasando por un momento de tensión o negatividad." \
                  " Tómate un momento para respirar profundo. Afirmación: *'Libero todo lo que no me sirve.'*"
    else:
        emotion = "NEUTRAL"
        message = "😐 Tu estado emocional es neutro. Es un buen momento para tomar conciencia." \
                  " Afirmación: *'Estoy presente y receptiva a lo nuevo.'*"

    st.success(f"Emoción estimada: {emotion}")
    st.info(message)
