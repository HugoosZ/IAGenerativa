from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch 


# Verificaciones finales: Detectar dispositivo disponible
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Paso 1: Cargar el archivo de correos
ruta_absoluta = "/Users/hugo/workspace/Practica1/IAGenerativa/PrimerosPasos/correos_generados.txt"
dataset = load_dataset("text", data_files={"train": ruta_absoluta})

# Paso 2: Tokenizar los datos
tokenizer = AutoTokenizer.from_pretrained("gpt2") #Primero se reconoce con que modelo vamos a trabajar de manera automatica
tokenizer.pad_token = tokenizer.eos_token #GPT-2 no tiene un token de padding(ajustar las longitudes de las secuencias uniformememente) por defecto, por lo que asignamos pad_token al mismo valor que eos_toke
print(f"PAD Token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"], 
        truncation=True,     # Trunca secuencias largas
        padding="max_length",  # Añade padding a las secuencias más cortas
        max_length=40         # Define una longitud fija para las secuencias
    )
    tokens["labels"] = tokens["input_ids"].copy()   #Copiar input_ids como labels
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True) # Se mandan los datos a la funcion tokenize_function por lotes
print(tokenized_dataset["train"][0])
#Paso 3: Cargar el modelo preentrenado
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

# Paso 4: Configurar el entrenamiento
training_args = TrainingArguments(
    output_dir="./results", # Carpeta donde se guardarán los resultados (modelo ajustado, checkpoints, etc.).
    overwrite_output_dir=True, # Sobrescribe la carpeta de resultados si ya existe.
    num_train_epochs=5,  # Número de Epoch (cuántas veces el modelo verá el conjunto de entrenamiento).
    per_device_train_batch_size=8, # Tamaño del batch que el modelo procesa simultáneamente.
    save_steps=500, # Guarda el modelo cada 500 pasos.
    save_total_limit=2, # Guarda solo los últimos 2 checkpoints (para ahorrar espacio).
    logging_dir="./logs",  # Carpeta donde se guardarán los registros (logs).
    logging_steps=50, # Cada 50 pasos se registra el progreso del entrenamiento.
)
# Paso 5: Entrenar el modelo
trainer = Trainer(
    model=model, # El modelo GPT-2 preentrenado.
    args=training_args, # Configuración del entrenamiento.
    train_dataset=tokenized_dataset["train"],  # Conjunto de datos de entrenamiento.
)
trainer.train()

#Paso 6: Generar texto con el modelo ajustado
def generar_texto(model, tokenizer, start_string, gen_length=100, temperature=0.7):
    input_ids = tokenizer.encode(start_string, return_tensors="pt").to(device) # Convierte el texto inicial  en una secuencia de IDs numéricos usando el tokenizador.
    output = model.generate(
        input_ids,
        max_length=gen_length,
        num_return_sequences=1,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

start_string = "correo"
generated_text = generar_texto(model, tokenizer, start_string, gen_length=100, temperature=0.7)
print("\nTexto generado:")
print(generated_text)
