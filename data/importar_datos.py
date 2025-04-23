import kagglehub
import os
import shutil  # Para mover o copiar archivos

# Descargar el dataset
path = kagglehub.dataset_download("dissfya/atp-tennis-2000-2023daily-pull")

# Ruta de destino para el archivo descargado en la carpeta raw
csv_path = os.path.join("raw")

# Mover el archivo descargado a la ubicación especificada en csv_path
shutil.move(path, csv_path)


print("Archivo movido a:", csv_path)
