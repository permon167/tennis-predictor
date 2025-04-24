
import pandas as pd

# Cargar el dataset
df = pd.read_csv("data/raw/atp_tennis.csv")

# Asegurarse que la columna Date sea de tipo fecha
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

# Filtrar desde el año 2021
df_filtered = df[df["Date"].dt.year >= 2021]

# Seleccionar las columnas relevantes
columns_needed = ["Player_1", "Player_2", "Surface", "Court", "Round", "Winner", "Score"]
df_filtered = df_filtered[columns_needed]

# Guardar a un nuevo CSV (opcional)
df_filtered.to_csv("data/processed/filtered.csv", index=False)
