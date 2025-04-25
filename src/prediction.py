import pandas as pd
import joblib

# Cargar el modelo entrenado
model = joblib.load("models/tennis_model_gradient_boosting.pkl")

# Cargar el dataset para obtener las estadísticas de los jugadores
df = pd.read_csv("data/processed/filtered.csv")

# Función para obtener las estadísticas de un jugador
def get_player_stats(player_name, surface):
    # Filtrar los datos por el jugador y la superficie
    player_data = df[((df["Player_1"] == player_name) | (df["Player_2"] == player_name)) & (df["Surface"] == surface)]
    
    # Asegurarse de que haya datos disponibles para este jugador y superficie
    if player_data.empty:
        print(f"No se encontraron datos para {player_name} en la superficie {surface}")
        return None, None, None, None, None
    
    # Obtener las estadísticas de los jugadores
    wins_surface = player_data["wins_surface_p1"].iloc[0] if player_name == player_data["Player_1"].iloc[0] else player_data["wins_surface_p2"].iloc[0]
    h2h_diff = player_data["h2h_diff"].iloc[0]  # Diferencia H2H
    rank_diff = player_data["rank_diff"].iloc[0]  # Diferencia de rankings (ya está calculada en el CSV)
    recent_form_p1 = player_data["recent_form_p1"].iloc[0]
    recent_form_p2 = player_data["recent_form_p2"].iloc[0]
    
    return wins_surface, h2h_diff, rank_diff, recent_form_p1, recent_form_p2

# Función de predicción
def predict_winner(player1, player2, surface):
    # Obtener las estadísticas de los dos jugadores
    stats_p1 = get_player_stats(player1, surface)
    stats_p2 = get_player_stats(player2, surface)
    
    if stats_p1[0] is None or stats_p2[0] is None:
        return "No se pudieron obtener los datos de los jugadores."

    # Crear un DataFrame con las características necesarias para la predicción
    new_data = {
        "wins_surface_p1": [stats_p1[0]],
        "wins_surface_p2": [stats_p2[0]],
        "h2h_diff": [stats_p1[1] - stats_p2[1]],  # Diferencia H2H entre los dos jugadores
        "rank_diff": [stats_p1[2] - stats_p2[2]],  # Diferencia de ranking entre los dos jugadores
        "recent_form_p1": [stats_p1[3]],
        "recent_form_p2": [stats_p2[3]],  # Nueva característica agregada
    }
    
    new_df = pd.DataFrame(new_data)

    # **Convertir las características en un array sin nombres de columna**
    X_pred = new_df.to_numpy()

    # Realizar la predicción
    prediction = model.predict(X_pred)
    predicted_class = "Jugador 1 Gana" if prediction[0] == 1 else "Jugador 2 Gana"
    
    return predicted_class


# Pedir entrada del usuario
player1 = input("Introduce el nombre del Jugador 1: ")
player2 = input("Introduce el nombre del Jugador 2: ")
surface = input("Introduce la superficie (ej. 'Hard', 'Clay', 'Grass'): ")

# Predecir el ganador
result = predict_winner(player1, player2, surface)
print(f"Predicción: {result}")



## Tenistas para probar: Sinner J., Fritz T., Ruud C., Rublev A., Djokovic N., Norrie C., Medvedev D.

#Alcaraz C.,Medvedev D.,6,1,5