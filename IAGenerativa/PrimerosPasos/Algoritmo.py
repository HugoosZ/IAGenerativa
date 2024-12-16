import random 

nombres = [
    "Juan", "Camila", "Pedro", "Sofía", "Valentina", "Martín", 
    "Diego", "Catalina", "Francisco", "Gabriela", "Sebastián", 
    "Javiera", "Carlos", "Antonia", "Ignacio", "María", "Felipe", 
    "Isabel", "Tomás", "Victoria", "Agustín", "Daniela", "Matías", 
    "Paula", "Andrés", "Fernanda", "Cristina", "Pablo", "Lorena", 
    "Rodrigo", "Carolina", "Luis", "Andrea", "Enrique", "Teresa"
]

apellidos = [
    "Pérez", "González", "Muñoz", "López", "Rodríguez", "Morales", 
    "Martínez", "Fernández", "Sánchez", "Ramírez", "Hernández", 
    "Castro", "Vargas", "Rojas", "Ortiz", "Reyes", "Jiménez", 
    "Torres", "Navarro", "Mendoza", "Guerrero", "Castillo", "Araya", 
    "Espinoza", "Vega", "Campos", "Carrasco", "Cortés", "Silva", 
    "Contreras", "Riquelme", "Pizarro", "Alvarado", "Cifuentes", 
    "Olivares", "Cáceres", "Valdés", "Salazar", "Figueroa", "Sepúlveda"
]

dominios = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", 
    "udp.cl", "usach.cl", "uchile.cl", "uc.cl", 
    "mail.com", "protonmail.com", "live.com", "icloud.com"
]


import unicodedata
def limpiador(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def generar_correo(C):
    correos = []
    for _ in range(C):
        nombre = random.choice(nombres).lower()
        apellido = random.choice(apellidos).lower()
        numero = random.randint(1,99)
        dom = random.choice(dominios).lower()

        correo = f"{nombre}.{apellido}{numero}@{dom}"
        correos.append(correo)
    return correos

def generador_correos_variados(C):
    separadores = ["", ".", "_", "-"]
    correos = []
    for _ in range(C):
        nombre = limpiador(random.choice(nombres).lower())
        apellido = limpiador(random.choice(apellidos).lower())
        sep = random.choice(separadores)
        numero = random.randint(1,99)
        dom = limpiador(random.choice(dominios).lower())
        if random.choice([True, False]):
            correo = f"{nombre}{sep}{apellido}{numero}@{dom}"
        else:
            correo = f"{nombre}{numero}@{dom}"
        correos.append(correo)
    return correos

def txt(correos_generados):
    with open("correos_generados.txt", "w") as archivo:
        for correo in correos_generados:
            archivo.write(correo + "\n")
    print("Listoko")

import csv
def csv_generator(correos_generados):
    with open("correos_generados.csv", "w", newline="", encoding="utf-8") as archivo:
        w = csv.writer(archivo)
        w.writerow(["Correo Electrónico"])  # Encabezado
        for correo in correos_generados:
            w.writerow([correo])
    print("ready el csv")

Cantidad = int(input())
correos_generados = generador_correos_variados(Cantidad)
for correo in correos_generados:
    print(correo)
txt(correos_generados)


