import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Cargar el dataset original
df = pd.read_csv("/Users/guadalupegarcia/Downloads/emotions.csv")

# Separar features y etiquetas
X = df.drop("label", axis=1)
y = df["label"]

# Preprocesamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Crear carpeta de modelo
os.makedirs("model", exist_ok=True)

# Guardar modelo y preprocesadores
joblib.dump(model, "model/random_forest_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoder, "model/label_encoder.pkl")

print("✅ Modelo entrenado y archivos guardados en /model")
