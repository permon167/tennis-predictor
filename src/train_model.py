import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint

# Cargar dataset
df = pd.read_csv("data/processed/filtered.csv").dropna()

# 🏆 Variable objetivo
y = (df["Winner"] == df["Player_1"]).astype(int)

# 🔍 Variables predictoras (eliminando Court_indoor y Round_num)
X = df[[ 
    "wins_surface_p1", 
    "wins_surface_p2", 
    "h2h_diff", 
    "rank_diff"
]]

# Normalización de las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🎯 Modelo: Gradient Boosting
model = GradientBoostingClassifier(random_state=42)

# **Ajuste de hiperparámetros con RandomizedSearchCV para Gradient Boosting**
param_dist = {
    'n_estimators': randint(100, 300),  # Prueba entre 100 y 300 estimadores
    'learning_rate': uniform(0.01, 0.1),  # Prueba entre 0.01 y 0.1
    'max_depth': randint(3, 6),  # Prueba entre 3 y 5
    'subsample': uniform(0.8, 0.2),  # Prueba entre 0.8 y 1.0
    'min_samples_split': randint(2, 10),  # Prueba entre 2 y 10
    'min_samples_leaf': randint(1, 5),  # Prueba entre 1 y 5
    'max_features': ['sqrt', 'log2', None]  # Prueba 'sqrt', 'log2' o None
}

# Búsqueda aleatoria para el modelo Gradient Boosting
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# **Entrenamiento con el mejor modelo encontrado**
best_model.fit(X_train, y_train)

# 🧠 Predicciones
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

print(f"✅ Accuracy: {acc:.4f}")
print(f"🎯 F1 Score: {f1:.4f}")
print(f"📈 ROC AUC: {roc:.4f}")

# 📊 Reporte
print("\n📊 Classification report:")
print(classification_report(y_test, y_pred, target_names=["Jugador 2 Gana", "Jugador 1 Gana"]))

# 📷 Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Jugador 2 Gana", "Jugador 1 Gana"], yticklabels=["Jugador 2 Gana", "Jugador 1 Gana"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
os.makedirs("models", exist_ok=True)
plt.savefig("models/confusion_matrix.png")
plt.close()
print("📷 Matriz de confusión guardada como: models/confusion_matrix.png")

# 🎯 Importancia de características
plt.figure(figsize=(8, 5))
feature_importance = best_model.feature_importances_
sns.barplot(x=feature_importance, y=X.columns)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("models/feature_importance.png")
plt.close()
print("📷 Importancia de características guardada como: models/feature_importance.png")

# 🔁 Cross-validation
cv_scores = cross_val_score(best_model, X_scaled, y, cv=3)  # Cambié a cv=3 para acelerar
print(f"🔁 Cross-validation accuracy (mean): {cv_scores.mean():.4f}")

# 💾 Guardar modelo
joblib.dump(best_model, "models/tennis_model_gradient_boosting.pkl")
print("📦 Modelo guardado en models/tennis_model_gradient_boosting.pkl")

# 📝 Guardar predicciones en un CSV para revisión
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Player_1_Ganador': (y_pred == 1).astype(int),  # Asumimos que 1 es jugador 1 gana
})
predictions_df.to_csv("models/predictions.csv", index=False)
print("📊 Predicciones guardadas en models/predictions.csv")
