from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch 


# Verificaciones finales: Detectar dispositivo disponible
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Usando dispositivo: {device}")



# Paso 1: Cargar el archivo de correos
ruta_absoluta = "/Users/hugo/workspace/Practica1/IAGenerativa/PrimerosPasos/correos_generados_mod.txt"
dataset = load_dataset("text", data_files={"train": ruta_absoluta})



# Paso 2: Tokenizar los datos
tokenizer = AutoTokenizer.from_pretrained("gpt2") #Primero se reconoce con que modelo vamos a trabajar de manera automatica
tokenizer.add_special_tokens({'pad_token': '[PAD]'}) #GPT-2 no tiene un token de padding(ajustar las longitudes de las secuencias uniformememente) por defecto, por lo que asignamos un token de padding explicito
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


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # No usamos "masked language modeling" porque es causal LM
)

#Paso 3: Cargar el modelo preentrenado
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

if tokenizer.vocab_size != len(tokenizer):
    original_device = model.device  # Guardar el dispositivo original
    model.to("cpu")  # Mover el modelo al CPU
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)  # Redimensionar embeddings
    model.to(original_device)  # Regresar el modelo al dispositivo original
    
model.config.pad_token_id = tokenizer.pad_token_id  # Configurar explicitamente pad_token_id


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
    data_collator=data_collator  # Se pasa el data_collator

)
trainer.train()

#Paso 6: Generar texto con el modelo ajustado
def generar_texto(model, tokenizer, start_string, gen_length=100, temperature=0.7, num_samples = 10):
    input_ids = tokenizer.encode(start_string, return_tensors="pt").to(device) # Convierte el texto inicial  en una secuencia de IDs numéricos usando el tokenizador.
    attention_mask = torch.ones_like(input_ids)  # Crear máscara de atención explícita87
    outputs = model.generate(
        input_ids,
        max_length=gen_length,
        num_return_sequences=num_samples,
        max_new_tokens=30,  # Generar hasta 30 tokens nuevos
        temperature=temperature,
        do_sample=True,  # Activar sampleo para usar temperature
        eos_token_id=tokenizer.eos_token_id,  # Usar token de fin de secuencia
        pad_token_id=tokenizer.pad_token_id,  # Incluido por seguridad
        early_stopping=True  # Detenerse cuando se encuentre <eos>
    )
    textos_generados = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return textos_generados

start_string = "correo"
num_samples = 20  # Número de correos a generar

generated_text = generar_texto(model, tokenizer, start_string, gen_length=30, temperature=0.7,num_samples=num_samples)
print("Correos generados:")
for i, texto in enumerate(generated_text, 1):
    print(f"{i}: {texto}")
