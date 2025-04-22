# Predictor de Resultados de Partidos de Tenis

Este proyecto tiene como objetivo desarrollar un modelo de aprendizaje automático que prediga los resultados de partidos de tenis basándose en estadísticas de los jugadores en partidos anteriores de los cuatro torneos Grand Slam: Abierto de Australia, Roland Garros, Wimbledon y US Open.

## Estructura del Proyecto

- **data/**: Contiene los archivos de datos en bruto y procesados.
  - **raw/**: Archivos de datos en bruto relacionados con estadísticas de jugadores y resultados de partidos.
  - **processed/**: Archivos de datos procesados que han sido limpiados y transformados para el análisis.
  
- **models/**: Contiene el modelo de aprendizaje automático entrenado.
  - **model.pkl**: El modelo entrenado utilizado para predecir los resultados de los partidos.
  
- **notebooks/**: Notebooks de Jupyter para análisis.
  - **exploratory_analysis.ipynb**: Notebook para análisis exploratorio de datos y visualizaciones.
  
- **src/**: Código fuente para el procesamiento de datos, ingeniería de características, entrenamiento del modelo y predicciones.
  - **data_preprocessing.py**: Funciones para cargar y preprocesar datos en bruto.
  - **feature_engineering.py**: Funciones para crear nuevas características que mejoren el rendimiento del modelo.
  - **model_training.py**: Código para entrenar el modelo de aprendizaje automático.
  - **prediction.py**: Funciones para realizar predicciones utilizando el modelo entrenado.
  
- **tests/**: Pruebas unitarias para el proyecto.
  - **test_data_preprocessing.py**: Pruebas unitarias para las funciones de preprocesamiento de datos.
  - **test_feature_engineering.py**: Pruebas unitarias para las funciones de ingeniería de características.
  - **test_model_training.py**: Pruebas unitarias para las funciones de entrenamiento del modelo.
  - **test_prediction.py**: Pruebas unitarias para las funciones de predicción.
  
- **requirements.txt**: Lista de dependencias necesarias para el proyecto.
- **.gitignore**: Especifica los archivos y directorios que deben ser ignorados por el control de versiones.
- **README.md**: Documentación del proyecto.

## Instrucciones de Configuración

1. Clona el repositorio:
   ```
   git clone <repository-url>
   cd tennis-predictor
   ```

2. Instala las dependencias necesarias:
   ```
   pip install -r requirements.txt
   ```

3. Prepara los datos:
   - Coloca los archivos de datos en bruto en el directorio `data/raw`.
   - Ejecuta el script de preprocesamiento de datos para limpiar y transformar los datos.

4. Entrena el modelo:
   - Utiliza el script `model_training.py` para entrenar el modelo de aprendizaje automático.

5. Realiza predicciones:
   - Utiliza el script `prediction.py` para realizar predicciones sobre nuevos datos de partidos.

## Guía de Uso

- Explora los datos utilizando el notebook de Jupyter en el directorio `notebooks`.
- Modifica el código fuente en el directorio `src` para mejorar el modelo o agregar nuevas características.
- Ejecuta las pruebas en el directorio `tests` para asegurarte de que el código funcione como se espera.

## Contribuciones

¡Las contribuciones son bienvenidas! Por favor, abre un issue o envía un pull request para cualquier mejora o corrección de errores.