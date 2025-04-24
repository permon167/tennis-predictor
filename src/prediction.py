import joblib
import pandas as pd

# Cargar modelo y codificadores
model = joblib.load("models/tennis_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# Datos nuevos para predecir (puedes cambiar estos valores)
input_data = {
    "Player_1": ["Djokovic N."],
    "Player_2": ["Alcaraz C."],
    "Surface": ["Hard"],
    "Court": ["Outdoor"],
    "Round": ["Final"]
}

# Crear DataFrame
df_input = pd.DataFrame(input_data)

# Codificar los datos igual que en entrenamiento
for col in df_input.columns:
    le = label_encoders[col]
    df_input[col] = le.transform(df_input[col])

# Hacer la predicción
pred_encoded = model.predict(df_input)[0]

# Decodificar resultado
winner = label_encoders["Winner"].inverse_transform([pred_encoded])[0]
print("El modelo predice que el ganador será:", winner)


## Tenistas para probar: Sinner J., Fritz T., Ruud C., Rublev A., Djokovic N., Norrie C., Nadal R., Federer R., Murray A.