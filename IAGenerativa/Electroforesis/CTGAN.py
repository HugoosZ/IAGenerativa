import pandas as pd
from ctgan import CTGAN
import os

# PASO 1: Cargar datos
ruta_csv = "/Users/hugo/workspace/Practica1/IAGenerativa/Electroforesis/dataset_tabular.csv"  # Ruta absoluta
datos = pd.read_csv(ruta_csv)

# PASO 2: Mostrar una vista rápida de los datos
print("Vista previa de los datos:")
print(datos.head())

# PASO 3: Especificar columnas categóricas
columnas_categoricas = ["Nombre", "Apellido", "Región", "Registro", "Correo Electrónico"]

# PASO 4: Inicializar el modelo CTGAN
modelo = CTGAN(
    epochs=300,  # Número de épocas (ajustable según rendimiento)
    batch_size=500,  # Tamaño del lote para entrenamiento
    generator_dim=(128, 128),  # Arquitectura del generador
    discriminator_dim=(128, 128),  # Arquitectura del discriminador
    verbose=True  # Para imprimir logs
)



# PASO 5: Entrenar el modelo
modelo.fit(datos, columnas_categoricas)

# PASO 6: Generar nuevos datos sintéticos
cantidad_filas = 80  # Número de filas a generar
datos_sinteticos = modelo.sample(cantidad_filas)

# PASO 7: Guardar los datos generados
ruta_salida = "/Users/hugo/workspace/Practica1/IAGenerativa/Electroforesis/datos_sinteticos.csv"
datos_sinteticos.to_csv(ruta_salida, index=False)

print(f"Nuevos datos sintéticos generados y guardados en {ruta_salida}")