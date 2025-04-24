import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar datos pre-filtrados
df = pd.read_csv("data/processed/filtered.csv")

# Eliminar filas con valores nulos
df = df.dropna()

# Codificar variables categóricas
label_encoders = {}
for col in ["Player_1", "Player_2", "Surface", "Court", "Round", "Winner"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separar características y etiqueta
X = df[["Player_1", "Player_2", "Surface", "Court", "Round"]]
y = df["Winner"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Guardar el modelo y los codificadores
import joblib
joblib.dump(model, "models/tennis_model.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")
