# Modificar el archivo de texto para incluir "<|endoftext|>" al final de cada correo
ruta_archivo = "/Users/hugo/workspace/Practica1/IAGenerativa/PrimerosPasos/correos_generados.txt"
ruta_archivo_modificado = "/Users/hugo/workspace/Practica1/IAGenerativa/PrimerosPasos/correos_generados_mod.txt"

with open(ruta_archivo, "r") as archivo:
    correos = archivo.readlines()

# Añadir "<|endoftext|>" al final de cada línea
correos_modificados = [correo.strip() + "<|endoftext|>\n" for correo in correos]

with open(ruta_archivo_modificado, "w") as archivo_modificado:
    archivo_modificado.writelines(correos_modificados)

print(f"Archivo modificado guardado en {ruta_archivo_modificado}")