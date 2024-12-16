import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Paso 1: Leer los datos desde un archivo
with open("correos_generados.txt", "r", encoding="utf-8") as file:
    data = file.read()

# Paso 2: Preprocesamiento de los datos
# Se toma el texto completo (data), identifica los caracteres únicos con set(data), los convierte en una lista, y los ordena alfabéticamente.
chars = sorted(list(set(data)))
char_to_index = {char: idx for idx, char in enumerate(chars)}
index_to_char = {idx: char for idx, char in enumerate(chars)}
# Paso 3: Convertir datos en índices
data_as_int = [char_to_index[char] for char in data] #Esta linea utiliza una list comprehension para crear una nueva lista. En este caso, la lista contiene los índices numéricos que corresponden a los caracteres en el texto data.



"""
La función preparar_datos toma un texto (data) y lo convierte en pares de entrada-salida 
para entrenar un modelo generativo. Cada entrada será una secuencia de caracteres, 
y cada salida será el siguiente carácter que sigue a esa secuencia.
"""
# Paso 4: Preparar secuencias de entrada y salida
def preparar_datos(data_as_int, seq_length=40): 

    # ej data : abcabc => [0, 1, 2, 0, 1, 2]
    X = []
    y = []
    for i in range(0, len(data_as_int) - seq_length):
        input_seq = data_as_int[i:i + seq_length] #slicing ; En un modelo generativo, queremos que el modelo aprenda la relación entre una secuencia (input_seq) y lo que viene inmediatamente después (target_char).
        target_char = data_as_int[i + seq_length] #ej: data_as_int = [0, 1, 2, 3, 4, 5, 6, 7] seq_length = 4
        #X = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]] y = [4, 5, 6, 7] entonces el primer espacio es el ultimo de lal segundo vector
        X.append(input_seq)
        y.append(target_char)
    X = np.array(X) 
    # X = [[0, 1, 2], [1, 2, 3], [2, 3, 4]] <=> X = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]) => Significa que X tiene 3 filas y 3 columnas, correspondiente a las 3 subsecuencias de entrada, cada una con 3 elementos.
    y = np.array(y)
    """
    ¿Por qué NumPy?
	Compatibilidad con modelos de Machine Learning:
	    - TensorFlow y otras bibliotecas esperan datos en formato de array de NumPy para entrenar modelos.
	    - Arrays tienen una estructura más eficiente para operaciones matemáticas en matrices.
    """    
    return X, y

seq_length = 40 # Elegimos 40 porque: Es suficientemente largo para capturar patrones útiles en los datos y no es demasiado largo como para aumentar excesivamente el tiempo de entrenamiento o la memoria requerida.
X, y = preparar_datos(data_as_int, seq_length)

# Paso 5: Construir el modelo
def construir_modelo(vocab_size, seq_length):
    model = Sequential() #Crea un modelo secuencial, que es una pila de capas donde la salida de cada capa es la entrada de la siguiente.
    model.add(Embedding(vocab_size, 256, input_length=seq_length)) #Embedding convierte los índices de caracteres de vectores densos a tamaño fijo.
    model.add(LSTM(256, return_sequences=True)) #Primera Capa Long Short-Term Memory el cual es ideal para datos secuenciales porque pueden “recordar” patrones en las secuencias a largo plazo.
    model.add(LSTM(256)) #	La primera capa captura patrones locales (por ejemplo, combinaciones comunes de caracteres como @gmail). y La segunda capa abstrae patrones más globales (por ejemplo, cómo los dominios interactúan con nombres).
    """
    ¿Por qué usar dos capas LSTM?
	    La primera LSTM captura patrones a nivel local (dentro de las secuencias).
	    La segunda LSTM abstrae patrones más complejos basados en la información procesada por la primera capa.
    """

    model.add(Dense(vocab_size, activation="softmax")) #La capa densa convierte la salida de la LSTM en probabilidades sobre el vocabulario (caracteres únicos).
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    """
        sparse_categorical_crossentropy:
            Es ideal porque nuestras etiquetas (y) son índices enteros, no vectores one-hot.
	    adam:
	        Un optimizador que ajusta automáticamente la tasa de aprendizaje.
	        Es rápido, robusto y generalmente funciona bien para este tipo de tareas.
    """

    return model


vocab_size = len(chars) #Total de caracteres únicos en el texto 
model = construir_modelo(vocab_size, seq_length)

#Paso 6: Entrenar el modelo
model.fit(X, y, epochs=20, batch_size=64, verbose=1) #epochs implica que el modelo pasará por los datos 10 veces y batch_size entrena en lotes de 64 secuencias para optimizar el uso de la memoria.
# verbose controla como se muestra la info, si es = 0, no se ve nada, = 1 muestra el progreso con metricas de perdida y exactitud, = 2, muetra metricas principales sin barra de progreso


#Paso 7: Generar texto
def generar_texto(model, start_string, gen_length=100,temperature=1.0):
    input_indices = [char_to_index[char] for char in start_string] #Convierte la cadena inicial start_string(Correo) en una lista de índices utilizando el diccionario char_to_index.
    input_indices = np.expand_dims(input_indices, axis=0) #Convierte input_indices en una matriz 2D para que sea compatible con el modelo. Esto se debe a que el modelo espera datos con una forma (batch_size, seq_length).
    generated_text = start_string #	El texto generado comienza con correo

    for _ in range(gen_length):
        predictions = model.predict(input_indices, verbose=0) #El modelo predice la probabilidad de cada carácter en el vocabulario para ser el siguiente carácter
        predictions = apply_temperature(predictions[0], temperature) # En lugar de np.argmax, usamos muestreo ponderado con np.random.choice, Esto permite que caracteres con menor probabilidad aún tengan alguna chance de ser seleccionados, reduciendo la repetitividad.
        predicted_index = np.random.choice(range(len(predictions)), p=predictions)

        predicted_char = index_to_char[predicted_index] #Usa el diccionario index_to_char para convertir el índice predicho en el carácter correspondiente.
        generated_text += predicted_char #Añade el carácter generado al texto.
        input_indices = np.roll(input_indices, -1) #Desplaza la secuencia una posición hacia la izquierda.
        input_indices[0, -1] = predicted_index #Sustituye el último índice por el índice predicho.

    return generated_text


def apply_temperature(predictions, temperature=1.0): #La temperatura ajusta la “confianza” de las probabilidades predichas, una temperatura más baja (e.g., 0.5) genera texto más “seguro” pero menos diverso.
    predictions = np.log(predictions + 1e-9) / temperature
    exp_preds = np.exp(predictions)
    return exp_preds / np.sum(exp_preds)

start_string = "correo"
generated_text = generar_texto(model, start_string, gen_length=100)
print("\nTexto generado:")
print(generated_text)