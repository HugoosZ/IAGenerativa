
Diferencias entre el modelo que se uso en PrimerosPasos y el modelo que se usará ahora

LSTM/RNN (Modelo TensorFlow anterior)
	•	Usa perceptrones y funciones de activación para procesar secuencias paso a paso.
	•	Es como “leer una novela palabra por palabra”, manteniendo una memoria limitada.
	•	La relación entre elementos lejanos en la secuencia se pierde rápidamente.

GPT-2 (Modelo actual)
	•	Usa un mecanismo de atención, específicamente “atención auto-regresiva”:
	•	Analiza todo el texto al mismo tiempo, evaluando la importancia de cada palabra respecto a las demás.
	•	No tiene un límite práctico en la memoria de contexto (puede aprender relaciones complejas entre palabras).
	•	Mucho más escalable y eficiente para secuencias largas.


Paso 2: Tokenizar los datos

    La función map en Hugging Face aplica una función (en este caso tokenize_function) a cada elemento del dataset. Osea aplica la funcion tokenize_function a cada elemento de dataset, además batched=True controla cómo se pasan los datos a la función, si es true, lo pasa por lotes. En la funcion quedará como {text: "Correo1", "Correo2"....}

Paso 3: Cargar el modelo preentrenado

    Se selecciona el modelo de manera automatica para tareas de lenguaje casual(Causal significa que el modelo predice el siguiente token (carácter/palabra) basándose en los tokens anteriores)

Paso 5: Configuracion de entrenamiento

    Aquí estamos configurando cómo se va a entrenar el modelo mediante el objeto Trainer.
        -Trainer es una clase de Hugging Face que simplifica el proceso de entrenamiento.
    En lugar de escribir todo el código manualmente para entrenar un modelo, Trainer automatiza:
	    •	La preparación de los datos.
	    •	El cálculo de la pérdida (loss).
	    •	El ajuste de pesos del modelo usando optimizadores.
	    •	La visualización de métricas y registros.
	    •	El guardado de checkpoints.
    Parámetros del Trainer:
	1.	model:
	    •	Es el modelo preentrenado que cargamos en el paso anterior (GPT-2).
	2.	args:
	    •	Son los argumentos de entrenamiento que definimos con TrainingArguments.
	3.	train_dataset:
	    •	Es el dataset tokenizado que vamos a usar para entrenar el modelo.

Paso 6: Generar texto con el modelo ajustado

	Primero se convierte el texto inicial (start_string) en una secuencia de IDs numéricos usando el tokenizador y se usa return_tensors="pt" para devolver los IDs como tensores de PyTorch, que son compatibles con el modelo. Luego se configura el output donde	input_ids es el contexto inicial como tensor, max_length=gen_length es la longitud máxima del texto generado (incluye el texto inicial), num_return_sequences=1 Genera una sola secuencia de salida, temperature=temperature introduce aleatoriedad en la generación. y pad_token_id=tokenizer.eos_token_id asegura que el modelo use el token de fin de secuencia (<|endoftext|>) para rellenar si es necesario.
	Finalmente se retorna la salida donde se convierte los IDs generados por el modelo (output[0]) de nuevo a texto legible y skip_special_tokens=True omite tokens especiales como <|endoftext|> o [PAD].


Como verificar que esta todo bien y el modelo va aprendiendo:

	- Si el loss(pérdida) disminuye consistentemente; en mi primera ejecucion disminuyo desde 1.07 a 0.29, lo que indica que el modelo va mejorando su capacidad para predecir correctamente el texto del dataset.

Por otra parte learning_rate, durante el entrenamiento, la tasa de aprendizaje disminuye gradualmente:

	-	Inicio: learning_rate = 4.96e-05
	-	Final: learning_rate = 0.0
Esto es un buen indicador de que se usó una estrategia de decaimiento del learning rate, lo cual ayuda a estabilizar el aprendizaje al final.

Ademas el gradiente norm(grad_norm) mide la magnitud de los gradientes usados para ajustar los pesos del modelo, donde: 
	-	Gradientes altos significan que el modelo está aprendiendo rápido, pero pueden ser inestables.
	-	Gradientes bajos indican que el modelo está ajustando de manera más controlada.


Finalmente, al ejecutar por primera vez, me salio este error:

 "RuntimeError: Placeholder storage has not been allocated on MPS device!"

Se debe a que PyTorch está intentando ejecutar operaciones en un dispositivo **MPS** (Metal Performance Shaders), utilizado por las GPU de Apple, pero los datos de entrada (input_ids) están en CPU. Esto genera una incompatibilidad.
Para solucionarlo, Debemos asegurarnos de que tanto el modelo como los datos estén en el mismo dispositivo (CPU o GPU). Esto implica mover los tensores de entrada y el modelo al dispositivo MPS si deseas aprovechar tu GPU. Por lo que al inicio del codigo, se importo torch y se detecta el dispositivo disponible, si esta MPS habilitado, lo usará, si no, usara CPU. Tanto como en la inicializacion del modelo, como cuando se genera el texto, se agregó la funcion ".to(device)" para que el modelo este en el dispositivo seleccionado y se asegure de mover los tensores generados por el tokenizador al dispositivo antes de pasarlos al modelo.

