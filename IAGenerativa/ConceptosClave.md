 Redes Neuronales Generativas

	•	IA Generativa: Modelos que pueden crear datos nuevos similares a los datos de entrada (como imágenes, texto, música).
	•	Aplicaciones comunes:
        Generación de texto (GPT, ChatGPT).
        Generación de imágenes (DALL-E, Stable Diffusion).
        Creación de datos sintéticos para entrenar otros modelos.

Conceptos básicos de IA generativa

	•	Entrenamiento supervisado vs. no supervisado:
	    •	En modelos generativos, muchas veces los datos no tienen “etiquetas” como en clasificación tradicional.
	    •	Ejemplo: Para generación de texto, los datos consisten en grandes conjuntos de palabras o caracteres.
	•	Codificación de datos (Embedding):
	    •	Transformar palabras o caracteres en vectores densos que representan sus relaciones semánticas.
	    •	Ejemplo: Palabras similares como “perro” y “gato” tendrán representaciones cercanas en el espacio vectorial.
	•	Tokenización:
	    •	Proceso de dividir un texto en unidades más pequeñas, llamadas tokens (pueden ser palabras, caracteres o partes de palabras).
	    •	Tokenizer: Herramienta que transforma texto en tokens y viceversa.

Modelos clave en IA generativa

	•	RNN (Redes Neuronales Recurrentes):
	    •	Diseñadas para manejar datos secuenciales.
	    •	Limitación: No son ideales para aprender dependencias a largo plazo.
	•	LSTM (Long Short-Term Memory):
	    •	Una mejora de las RNN que puede “recordar” información durante más tiempo.
	•	Transformers:
	    •	Reemplazan las RNN/LSTM en muchos casos.
	    •	Basados en la atención (self-attention), lo que les permite manejar dependencias a largo plazo de manera más eficiente.
	•	Modelos como GPT:
	    •	GPT (Generative Pre-trained Transformer) se basa en transformers y se entrena en grandes cantidades de texto para generar contenido coherente.

PyTorch

	•	Tensores:
	    •	Estructura de datos base en PyTorch, similar a matrices de Numpy pero con soporte para operaciones en GPU.
	•	Redes neuronales:
	    •	Construcción modular con torch.nn y torch.optim para crear y entrenar modelos.
	•	Autograd:
	    •	Sistema automático de cálculo de gradientes. Esencial para ajustar los pesos durante el entrenamiento.

Hugging Face
	•	Transformers Library:
	    •	Biblioteca que permite cargar modelos avanzados (GPT, BERT, etc.) para tareas como generación de texto o clasificación.
	•	Datasets:
	    •	Repositorios de datos para entrenar modelos de manera eficiente.
	•	Modelos preentrenados:
	    •	Modelos ya entrenados en grandes conjuntos de datos, listos para ajustar con datos específicos.

Procesamiento de texto

	•	Secuencias de entrada y salida:
	    •	Crear pares de datos (X, y) donde X es una secuencia de texto y y es el carácter o palabra siguiente.
	•	One-hot encoding vs. Embedding:
	    •	One-hot encoding: Representa cada token como un vector esparcido con un único “1”.
	    •	Embedding: Representa tokens como vectores densos y continuos.

Loss Functions (Funciones de Pérdida)

	•	Sparse Categorical Crossentropy:
	    •	Evalúa qué tan bien predice el modelo el próximo token en una secuencia.
	•	Temperature:
	    •	Controla la aleatoriedad durante la generación de texto.
	    •	Valores bajos hacen que el modelo sea más predecible, mientras que valores altos aumentan la creatividad.


Aplicaciones prácticas

	•	Generación de texto:
	    •	Entrenar modelos para completar oraciones, generar correos electrónicos o crear descripciones.
	•	Fine-tuning (ajuste fino):
	    •	Adaptar un modelo preentrenado a un conjunto de datos específico.

 Flujo típico de un modelo generativo
 
	1.	Preprocesamiento:
	    •	Limpiar, normalizar y transformar los datos en un formato utilizable por el modelo.
	2.	Entrenamiento:
	    •	Alimentar datos secuenciales para que el modelo aprenda patrones.
	3.	Generación:
	    •	Usar el modelo para crear nuevas secuencias (texto, datos, etc.).
	4.	Evaluación:
	    •	Medir qué tan realistas o útiles son las secuencias generadas.