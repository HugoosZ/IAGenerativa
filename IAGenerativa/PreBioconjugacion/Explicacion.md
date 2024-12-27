
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

Primer Print: 

	UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.7` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
	warnings.warn(
	The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

	Texto generado:
	correo-carrasco99@protonmail.com

Se demoro casi 3 horas en compilar en mi macbook air M1. Le agregue do_sample=True para activar el uso de temperatura, que por defecto viene desactivado. Ademas, para resolver el tema del attention_mask, le agregué attention_mask = torch.ones_like(input_ids) en la funcion de generacion de texto para crear una mascara explicita a inputs_id, asi el modelo sabrá qué tokens debe considerar y cuáles debe ignorar; donde 	
	•	1: Tokens válidos (parte del input real).
	•	0: Tokens de relleno (tokens pad).

Segunda Ejecucion:

The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

Texto generado:
correo.olivares44@usach.cl

Para solucionarlo, se quito el eos.token como pad_token, porque cuando ambos tokens son iguales, el modelo no puede distinguir entre el token que termina una secuencia válida (eos) y el token que solo rellena para mantener el tamaño fijo (pad). Por otra parte se agrego dataCollector, el cual calcula el padding dinámicamente al tamaño de la secuencia más larga en cada lote.

Finalmente, antes de compilar, se ajustará el codigo para hacer mas correos, agregando el parametro a la funcion generar texto: num_samples, y este se asignarpa en el modelo del output. El return igual se verá afectado debido a que ya no es solo un correo, por lo que se tendrá que recorrer outputs.

Tercera ejecucucion:


The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

Correos generados:
	1: correo.olivares44@usach.cl.cl.com.clonia.cl.com.com.com.comicifuentes6@mail.com.com.com.com.clonia.cl.usaula.cast.com.clasco.guerrero.com.contreras11@uc.cl.com.com.uchile.cl.cl.closes.clos.clovega29
	2: correo.espinoza28@gmail.com.com.com.com.com.uchile.clasco.cl.clarro.closes.usach.clare.clos67@live.com.com.cl.com.is.clonia.clares.com.live.com.loud.contreras68@gmail.com.com.com.cl.usach.closcifuentes45@us
	3: correo_sepulveda78@icloud.com.com.com.com.protonmail.com.com.com.uc.clos.uchile.clos20@protonmail.com.com.com.com.munoz.com.com.is8@protonmail.com.com.com.com.com.comentes.protonmail.comorales67@protonmail.com.
	4: correo_espinoza46@protonmail.com.com.com.com.com.com.comcast.com.com.sanchez51@icloud.com.com.com.com.com.com.cast.com.ro.cifuentes37@live.com.com.com.com.uc.clasco97@udp.cl.cl.cl.clifres.cl.clifro
	5: correo82@mail.com.com.com.com.com.com.silva.usach.clove.com.sanchez68@usach.cl.com.cl.com.com.clare.cl.clare.cl.usach.clos.clove.usaches35@protonmail.com.com.uchile.clare.usach.clos2@mail.com.comuchile.
	6: correo.cifuentes92@gmail.com.com.com.com.uc.clasco.cl.com.clare.clasco43@udp.cliv.clonia.clariquel.cl.cast.com.com.clifuentes40@uchile.cl.clos.cl.cl.clare.cl.uc.clareyes17@outlook.com.com.uc
	7: correo_martinez46@yahoo.comcast.comcast.com.com.com.cast.comcastro.com.live.com.comcastro.comcastro.castillo35@icloud.com.com.comcast.comcast.com.comam+(mail.com.comprotonmail.com.com.hernandez56@usach.cl.cl.cl.cl.com.com.usach
	8: correo93@udp.cl.cl.com.com.com.com.com.closes.comcast.com.com.comcastro37@udp.cl.cl.cl.comcast.com.clasco.co.us.clos.live.comcastrojas61@udp.clas.cl.cl.cloriquel.clifuentes.com.cl@hotmail.com.
	9: correo-munoz11@hotmail.com.com.com.com.com.uc.clasco.closco21@live.com.com.com.uchile.clofia.closarro21@hotmail.com.com.com.usach.cl.cl.clos_torres68@uchile.com.cloriquelme.comudp.closcifuentes_silva72
	10: correo.castro64@gmail.com.com.com.com.com.comortach.cl.protonmail.com.castrojas23@yahoo.com.com.com.com.com.comcast.com.com-contreras.comorres.cl.comga88@gmail.com.com.com.com.protonmail.comicloud.comcastro20@hotmail.com.com
	11: correo_contreras28@uc.clar.clas.cl.clos.com.com.com.clos.uc.clos.cloz.clifuentes45@live.com.comcast.com.com.usach.cl.clasco.live.com.com.argas86@udp.clar.closc.cl.clismorales.clerriare.comcastro
	12: correo.castillo47@uc.clar.cl.clonia.cl.com.com.com.com.comcast.comcast.com.p.clarro99-silva20@uc.clar.cl.clonia.clos.cl.com.com.comcast.comcastro32@hotmail.com.com.com.comcast.com.com.com.torresanchez10@protonmail.
	13: correo.lopez59@uchile.cl.cl.clas.cl.cl.protonmail.com.com.com.com.aula.clove.contreras.clasco.usach.clare.clare.clareme73@yahoo.com.com.clos.com.com.protonmail.com.com.cast.com.protonmail.comorales85@hotmail.com
	14: correo.caceres78@protonmail.com.com.comcast.com.com.com.sanchez16@outlook.com.com.com.com.comcast.comcast.com.comcastro.contreras85@protonmail.com.com.usach.clariquelme53@udp.clare.com.com.asprotonmail.comprotonmail.com.com.
	15: correo92@mail.com.com.com.com.com.aro.com.clsanchez34@yahoo.com.com.com.com.com.cast.com.com.com.clas.rojas.cl.rojas.castillo39@yahoo.com.com.com.com.com.castrojas.uchile.loud.com.usach.rojas68@uc.clare.clar.
	16: correo78@protonmail.com.com.com.com.clas.com.comcast.com.com.cast.comorales.cl.uchonzalez29@icloud.com.cl.com.comcast.com.cast.comos23@uc.clar.cl.clove.com.comcastirez28@hotmail.com.com.com.protonmail.com.com-contreras
	17: correo.carrasco46@gmail.com.com.com.com.com.com.usach.cl.cl.protonmail.com.comusach.cl.cl.clifueroa28@uc.cl.cl.cl.cl.cl.com.uc.clasco.uchile.closcifres.torres93@uc.cl.cl.cl.ocace.cl.cl.oc
	18: correo27@protonmail.com.com.com.com.comcast.com.comcast.com.comcast.usach.cl.cl.ro.rojas90@uc.clar.clos.cl.cl.cl.ocastro.closcifuero_live.com.com.sanchez26@mail.com.com.clos.cl.com.gonzalez28@usach.cl.
	19: correo11@usach.cl.com.clove.cl.com.com.comas.ucarrasco.live.comorales97@uchile.clonia.cls.cl.cler.com.usarlos.clonzalez40@usach.cl.clar.cl.clar.cl.clove.clore.clare.com.torres31@usach.cl.clifuent
	20: correo15@yahoo.com.com.com.com.com.cast.cast.comcast.com.ro.loud.comro.alvarado.clareyes52@udp.cl.cl.cl.cl.usach.clar.closcil.clifuentes32@gmail.com.com.com.com.com.-@uc.cl.clerrero.clarero23@yahoo.

	El problema del padding persiste, asi que configuré diferentes Tokens de Padding, 



	Cuarta ejecucion:  to 1. However, `early_stopping` is set to `True` -- this flag is only used in beam-based generation modes. You should set `num_beams>1` or unset `early_stopping`.
  warnings.warn(
Both `max_new_tokens` (=30) and `max_length`(=30) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
Correos generados:
1: correo.olivares44@usach.cl
2: correo.reyes46@yahoo.com
3: correo-silva32@hotmail.com
4: correo-ortiz33@outlook.com
5: correo95@udp.cl
6: correo13@protonmail.com
7: correo-olivares18@live.com
8: correo_gonzalez32@udp.cl
9: correo.carrasco80@udp.cl
10: correo68@uchile.cl
11: correo78@outlook.com
12: correo78@yahoo.com
13: correo-cifuentes9@live.com
14: correo.caceres93@gmail.com
15: correo-alvarado94@live.com
16: correo_castillo83@live.com
17: correo_contreras63@hotmail.com
18: correo-caceres97@uchile.cl
19: correo81@live.com
20: correo.cifuentes5@udp.cl 
Investigando, me enteré el que archivo que se usa para entrenar la IA, tiene que tener al final de cada linea "<|endoftext|>", debido a que GPT2 no sabe donde termina cada linea, asi podemos evitar el overfitting.