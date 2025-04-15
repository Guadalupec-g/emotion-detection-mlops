# 📊 Información del Modelo de Detección de Emociones

## 🎯 Variables de entrada del modelo

- Total de variables: **2548 columnas numéricas**
- Formato: CSV sin encabezado de etiquetas
- Contenido: señales EEG procesadas y transformadas (features extraídas del cerebro)
- Preprocesamiento aplicado:
  - Escalado con `StandardScaler` de Scikit-learn

## 🧠 Algoritmo

- Modelo: `RandomForestClassifier`
- Framework: `Scikit-learn`
- Entrenamiento automático mediante GitHub Actions (`train_and_save_model.py`)

## 📤 Salida del modelo

- Emoción predicha: una de las siguientes clases:
  - `POSITIVE`
  - `NEUTRAL`
  - `NEGATIVE`
- Tipo de salida: etiqueta de clase (`str`)
- Codificación interna con `LabelEncoder`:
  - `0 = NEGATIVE`
  - `1 = NEUTRAL`
  - `2 = POSITIVE`

## 🗃️ Archivos generados

- `model/random_forest_model.pkl`
- `model/scaler.pkl`
- `model/label_encoder.pkl`
