
Primero busco entender la IA Generativa mediante CorreosIA.py, donde analizaré que hace cada funcion detalladamente. Debido a que no se mucho sobre Py, primero creé Algoritmo.py, el cual generá correos aleatorios bajo una cantidad de nombres, apellidos y dominios determinados. Finalmente, se cargarán estos correos a CorreosIA.py los cuales se usarán para poder entrenar el modelo.

Pasos de correos IA

Paso 2: Preprocesar los Datos

    Supongamos que el archivo contiene: hola mundo
    entonces data = "hola mundo"
    set divide cada caracter y elimina duplicados, asi: {h, o, l, a, m, u, n, d, ' '} luego list lo hace una lista, dejando: ['h', 'o', 'l', 'a', 'm', 'u', 'n', 'd', ' ']
    y luego se ordena mediante sorted, dejandolo [' ', 'd', 'h', 'l', 'm', 'n', 'o', 'u']
    luego char to index genera un diccionario de indices, asignandole un numero a cada caracter
    char_to_index = {' ': 0, 'd': 1, 'h': 2, 'l': 3, 'm': 4, 'n': 5, 'o': 6, 'u': 7}
    e index_to_char, hace un diccionario inverso

Paso 3: Convertir Texto a Índices

    Convierte el texto completo (data) en una lista de números (índices) usando char_to_index. con "hola mundo" 
    data_as_int = [2, 6, 3, 3, 0, 4, 7, 5, 1, 6]. Donde:

        - 'h' → 2
        - 'o' → 6
        - 'l' → 3
        - ' ' → 0 (espacio)
        - Y así sucesivamente.

Paso 4: Preparar Secuencias de Entrada y Salida

    Entrada (X):
	    Subcadenas de longitud seq_length extraídas de data_as_int.
	Salida (y):
	    El carácter siguiente a cada subsecuencia en X.

    Ejemplo:

    Con data_as_int = [2, 6, 3, 3, 0, 4, 7, 5, 1, 6] y seq_length = 4:

    Iteraciones:
        1.	Primera iteración (i = 0):
        •	input_seq = [2, 6, 3, 3] → Corresponde a "hola".
        •	target_char = 0 → Corresponde al espacio ' '.
        2.	Segunda iteración (i = 1):
        •	input_seq = [6, 3, 3, 0] → Corresponde a "ola "
        •	target_char = 4 → Corresponde a 'm'.
        y asi...
    Resultado:
   
            X = [
            [2, 6, 3, 3],   "hola"
            [6, 3, 3, 0],   "ola "
            [3, 3, 0, 4],   "la m"
            [3, 0, 4, 7],   "a mu"
            [0, 4, 7, 5],   " mun"
            [4, 7, 5, 1],   "mund"
        ]
        y = [0, 4, 7, 5, 1, 6] → [' ', 'm', 'u', 'n', 'd', 'o']

Paso 5: Cronstruccion del modelo:
    Importante!! Un vector denso es un arreglo de números reales (flotantes) que contiene información distribuida en todas sus dimensiones.

    Primero creamos un modelo secuencial donde cada capa es apilada una tras otra. donde luego se convierte los índices de los caracteres en vectores densos de 256 dimensiones. y cada parámetro representa:
	    vocab_size: Número de caracteres únicos en el texto.
	    256: Tamaño de los vectores embedding.
	    input_length=seq_length: Longitud fija de las secuencias de entrada (40).
    Luego, la tercera linea , crea una capa que procesa secuencias de entrada y devuelve una salida para cada paso temporal, donde cada parametro representa:
        256: Número de unidades (dimensión de salida).
	    return_sequences=True: Asegura que la capa devuelva una salida por cada paso temporal.
    Se agrega una segunda capa en la siguiente linea, la cual procesa la secuencia completa y devuelve solo la salida final. Donde solo tiene 256 como parametro y esta sera la dimesion de salida la cual no tiene return_sequences=True, porque esta es la última capa recurrente. Solo necesitamos la salida final de la secuencia (no todos los pasos).
    Finalmente en model.add se predice la probabilidad de cada carácter en el vocabulario como el siguiente carácter en la secuencia. y los parametros 
        vocab_size: Número de caracteres únicos en el texto.
	    activation="softmax": Convierte las salidas en probabilidades.
    y se compilará el modelo bajo sparse_categorical_crossentropy, adam, accuracy. Donde cada uno define cómo se mide el error del modelo, cómo se ajustan los pesos del modelo para reducir el error, las métricas que se mostrarán durante el entrenamiento, Respectivamente.


Paso 7: Generacion de texto

    En esta etapa, el modelo utiliza lo aprendido durante el entrenamiento para generar texto en base a una cadena inicial (start_string). El proceso comienza convirtiendo la cadena inicial en índices numéricos utilizando el diccionario char_to_index. A partir de estos índices, el modelo predice el siguiente carácter con la mayor probabilidad utilizando np.argmax, que selecciona la opción más probable directamente. Este carácter predicho se agrega al texto generado y reemplaza la posición más antigua en la secuencia de entrada, actualizando así los índices que el modelo utiliza para la siguiente predicción. El bucle continúa hasta alcanzar la longitud deseada del texto (gen_length). Este método asegura que el texto siga patrones consistentes aprendidos durante el entrenamiento, pero debido a la falta de variabilidad, puede resultar en repeticiones en ciertos contextos.

Primera Generacion de texto: 

correo22@uchile.castro28@uchile.castro28@uchile.castro28@uchile.castro28@uchile.castro28@uchile.castro28@u.


Segun GPT: Por qué el texto generado es repetitivo?

Esto sucede por varios motivos posibles:
	1.	Datos de entrenamiento limitados o sesgados:
	•	Si los datos de entrenamiento contienen patrones muy repetitivos, el modelo tenderá a generar esas mismas repeticiones.
	2.	Predicción determinista (np.argmax):
	•	Usar np.argmax selecciona siempre el carácter más probable, lo que puede llevar a repeticiones constantes.
	•	Para evitar esto, podemos introducir aleatoriedad controlada en la generación del texto.
	3.	Sobreajuste:
	•	A medida que el modelo memoriza los datos, puede ser menos creativo al generar texto.


Segunda Generacion de texto: Le agregué la funcion apply_temperature para poder generar patrones mas variados y no basados en el caracter que tiene la mas alta posibilidad de continuar.

correo81@icloudez7@gmail.contrerascaldva53@protonrrez90@uc.cl
enia29@icloudez27@icloudee1cprerodriguez25@m

Tercera Generacion con 20 Epoch:

correo36@udp.clgres34@hotmaira.morales10@outloolia11@hotmaira-rodriguez4@icloud46@hotmaiss23@icloud41@outl


Tips para poder mejorar el modelo: 

-seq_length = 50 puede ayudar al modelo a capturar patrones más amplios
-model.add(LSTM(256, return_sequences=True, dropout=0.2)) Agregar técnicas como Dropout al modelo puede ayudar a evitar sobreajuste y mejorar la capacidad de generalización
-Ajustar la temperatura puede ayudar a reducir los errores en las salidas: temperature=0.8


Siguiente paso; usar modelos de HugginFace para poder generar datos en la carpeta PreBioconjugacion

