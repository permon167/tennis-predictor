import pandas as pd
import os

def procesar_datos_us_open():
    """Procesa el archivo CSV y filtra las columnas relacionadas con el US Open."""
    # Ruta del archivo original en el directorio raw
    raw_csv_path = os.path.join("raw", "atp_tennis.csv")
    
    # Ruta del archivo procesado en el directorio processed
    processed_csv_path = os.path.join("processed", "us_open_data.csv")
    
    # Cargar los datos desde el archivo CSV
    try:
        data = pd.read_csv(raw_csv_path)
    except FileNotFoundError:
        print(f"El archivo {raw_csv_path} no existe.")
        return
    
    # Filtrar las filas y columnas relacionadas con el US Open
    # Supongamos que hay una columna llamada "tournament" que identifica el torneo
    us_open_data = data[data["Tournament"] == "US Open"]

     # Filtrar las filas desde el año 2020
    us_open_data = us_open_data[us_open_data["Date"] >= "2021-01-01"]
    
    # Seleccionar columnas relevantes (ajusta según las columnas disponibles)
    columnas_relevantes = ["Date", "Player_1", "Player_2", "Score","Winner", "Round", "Surface"]
    us_open_data = us_open_data[columnas_relevantes]
    
    # Crear el directorio processed si no existe
    os.makedirs(os.path.dirname(processed_csv_path), exist_ok=True)
    
    # Guardar los datos procesados en el archivo CSV
    us_open_data.to_csv(processed_csv_path, index=False)
    print(f"Datos procesados guardados en: {processed_csv_path}")

if __name__ == "__main__":
    procesar_datos_us_open()