import random
import unicodedata
import csv
from datetime import datetime, timedelta

# Listas base
nombres = ["Juan", "Camila", "Pedro", "Sofía", "Valentina", "Martín", "Diego", "Catalina"]
apellidos = ["Pérez", "González", "Muñoz", "López", "Rodríguez", "Morales", "Martínez", "Fernández"]
dominios = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "uchile.cl", "udp.cl"]
regiones = ["Santiago", "Valparaíso", "Antofagasta", "Concepción", "La Serena"]

# Limpieza de texto
def limpiador(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

# Generador de correos electrónicos
def generar_correo(nombre, apellido):
    nombre = limpiador(nombre.lower())
    apellido = limpiador(apellido.lower())
    numero = random.randint(1, 99)
    dom = random.choice(dominios)
    sep = random.choice(["", ".", "_", "-"])
    return f"{nombre}{sep}{apellido}{numero}@{dom}"

# Generador de fechas aleatorias
def generar_fecha():
    inicio = datetime(2020, 1, 1)
    fin = datetime(2023, 12, 31)
    return inicio + timedelta(days=random.randint(0, (fin - inicio).days))

# Generar dataset tabular
def generar_dataset_tabular(C):
    filas = []
    for _ in range(C):
        nombre = random.choice(nombres)
        apellido = random.choice(apellidos)
        edad = random.randint(18, 65)
        region = random.choice(regiones)
        fecha_registro = generar_fecha().strftime("%Y-%m-%d")
        correo = generar_correo(nombre, apellido)

        filas.append([nombre, apellido, edad, region, fecha_registro, correo])
    return filas

# Guardar en CSV
def guardar_csv(filas, nombre_archivo="dataset_tabular.csv"):
    with open(nombre_archivo, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["Nombre", "Apellido", "Edad", "Región", "Registro", "Correo Electrónico"])  # Encabezado
        escritor.writerows(filas)
    print(f"Archivo guardado: {nombre_archivo}")

# Configuración principal
Cantidad = int(input("Cantidad de filas a generar: "))
dataset = generar_dataset_tabular(Cantidad)
guardar_csv(dataset)