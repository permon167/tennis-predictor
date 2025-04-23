import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar los datos procesados
data = pd.read_csv("data/processed/us_open_data.csv")

# Crear la columna 'Winner' (1 si Player_1 ganó, 0 si Player_2 ganó)
data["Winner"] = (data["Winner"] == data["Player_1"]).astype(int)

# Codificar variables categóricas
data = pd.get_dummies(data, columns=["Surface", "Round"], drop_first=True)

# Seleccionar características y el objetivo
X = data.drop(columns=["Winner", "Player_1", "Player_2", "Score", "Date"])
y = data["Winner"]

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Crear y entrenar el modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

# Crear un ejemplo de partido futuro
partido_futuro = {
    "Surface_Hard": 1,  # Superficie dura
    "Round_Final": 1,   # Ronda final
    # Agrega otras características relevantes
}

# Convertir a DataFrame
partido_futuro_df = pd.DataFrame([partido_futuro])

# Hacer la predicción
prediccion = model.predict(partido_futuro_df)
print("Ganador predicho:", "Player 1" if prediccion[0] == 1 else "Player 2")