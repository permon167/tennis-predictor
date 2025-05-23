import pandas as pd
from collections import defaultdict
import os

# Cargar datos
df = pd.read_csv("data/raw/atp_tennis.csv")
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

# Filtrar desde 2022 y eliminar filas con nulos clave
df = df[df["Date"].dt.year >= 2022].dropna(subset=["Player_1", "Player_2", "Winner", "Surface", "Court", "Round", "Date"])

# Contar partidos por jugador y limitar a los 100 jugadores más frecuentes
all_players = pd.concat([df["Player_1"], df["Player_2"]])
player_counts = all_players.value_counts()
valid_players = player_counts.head(100).index
df = df[df["Player_1"].isin(valid_players) & df["Player_2"].isin(valid_players)]

# Ordenar por fecha para evitar data leakage
df = df.sort_values("Date")

# Inicializar contadores
h2h_wins = defaultdict(lambda: [0, 0])
surface_wins = defaultdict(lambda: {"Player_1": 0, "Player_2": 0})
recent_form = defaultdict(list)  # Almacena los resultados recientes de cada jugador
h2h_diff_list = []
wins_surface_p1_list = []
wins_surface_p2_list = []
recent_form_p1_list = []
recent_form_p2_list = []

# Recorrer partidos y calcular stats dinámicas
for _, row in df.iterrows():
    p1, p2 = row["Player_1"], row["Player_2"]
    surface = row["Surface"]
    winner = row["Winner"]
    key = tuple(sorted([p1, p2]))
    p1_first = key[0] == p1

    # Head-to-head
    p1_wins, p2_wins = h2h_wins[key]
    h2h_diff = p1_wins - p2_wins if p1_first else p2_wins - p1_wins
    h2h_diff_list.append(h2h_diff)

    # Surface wins
    wins_p1 = surface_wins[surface].get(p1, 0)
    wins_p2 = surface_wins[surface].get(p2, 0)
    wins_surface_p1_list.append(wins_p1)
    wins_surface_p2_list.append(wins_p2)

    # Calcular rendimiento reciente
    p1_recent_wins = sum(recent_form[p1][-5:]) / min(len(recent_form[p1]), 5) if recent_form[p1] else 0
    p2_recent_wins = sum(recent_form[p2][-5:]) / min(len(recent_form[p2]), 5) if recent_form[p2] else 0
    recent_form_p1_list.append(p1_recent_wins)
    recent_form_p2_list.append(p2_recent_wins)

    # Actualizar contadores
    if winner == p1:
        h2h_wins[key][0 if p1_first else 1] += 1
        surface_wins[surface][p1] = wins_p1 + 1
    else:
        h2h_wins[key][1 if p1_first else 0] += 1
        surface_wins[surface][p2] = wins_p2 + 1

    # Actualizar resultados recientes
    recent_form[p1].append(1 if winner == p1 else 0)
    recent_form[p2].append(1 if winner == p2 else 0)

# Agregar columnas al DataFrame
df["h2h_diff"] = h2h_diff_list
df["wins_surface_p1"] = wins_surface_p1_list
df["wins_surface_p2"] = wins_surface_p2_list
df["recent_form_p1"] = recent_form_p1_list
df["recent_form_p2"] = recent_form_p2_list
df["rank_diff"] = df["Rank_2"] - df["Rank_1"]

# Codificar ronda como número
round_order = {
    "Round of 128": 1, "Round of 64": 2, "Round of 32": 3, "Round of 16": 4,
    "Quarterfinals": 5, "Semifinals": 6, "Final": 7
}
df["Round_num"] = df["Round"].map(round_order).fillna(0)

# Codificar tipo de cancha
df["Court_indoor"] = (df["Court"] == "Indoor").astype(int)

# Guardar columnas seleccionadas
cols = [
    "Date", "Player_1", "Player_2", "Rank_1", "Rank_2", "Surface",
    "wins_surface_p1", "wins_surface_p2", "h2h_diff",
    "rank_diff", "recent_form_p1", "recent_form_p2", "Winner"
]
df[cols].to_csv("data/processed/filtered.csv", index=False)

print("✅ Datos procesados guardados en data/processed/filtered.csv")

# Crear un DataFrame para los H2H
h2h_data = []

for (player1, player2), (wins_p1, wins_p2) in h2h_wins.items():
    h2h_data.append({
        "Player_1": player1,
        "Player_2": player2,
        "Wins_Player_1": wins_p1,
        "Wins_Player_2": wins_p2,
        "H2H_Diff": wins_p1 - wins_p2
    })

h2h_df = pd.DataFrame(h2h_data)

# Guardar el archivo CSV
h2h_df.to_csv("data/processed/h2h_stats.csv", index=False)

print("✅ Archivo H2H guardado en data/processed/h2h_stats.csv")
