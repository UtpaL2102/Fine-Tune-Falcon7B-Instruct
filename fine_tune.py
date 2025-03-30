import os
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # ✅ Fixed import
from datasets import load_from_disk

# ✅ Disable W&B logs
os.environ["WANDB_DISABLED"] = "true"

# ✅ Define model
model_name = "tiiuae/falcon-7b-instruct"

# ✅ Enable 4-bit quantization (ONLY if CUDA is available)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
) if torch.cuda.is_available() else None

# ✅ Load Model with Quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# ✅ Prepare model for LoRA + k-bit training
model = prepare_model_for_kbit_training(model)

# ✅ Disable caching for LoRA compatibility
model.config.use_cache = False

# ✅ Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Load Tokenized Dataset
dataset_path = "/content/tokenized_dataset"
tokenized_dataset = load_from_disk(dataset_path)

# ✅ Ensure Dataset Has Labels for CLM
def add_labels(example):
    example["labels"] = example["input_ids"].copy() if "input_ids" in example else None
    return example

train_dataset = tokenized_dataset.map(add_labels, remove_columns=["text"])

# ✅ Apply LoRA with Correct Target Modules for Falcon-7B
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value"],  # ✅ Falcon-7B uses "query_key_value"
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)

# ✅ Ensure LoRA Parameters are Trainable
for param in model.parameters():
    param.requires_grad = False  # Freeze all parameters

for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True  # Enable gradients for LoRA layers

# ✅ Move model to CUDA if available
if torch.cuda.is_available():
    model.to("cuda")

# ✅ Debug: Print Trainable Parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"✅ Trainable: {name}")
    else:
        print(f"❌ Frozen: {name}")

model.print_trainable_parameters()  # ✅ Verify trainable params

# ✅ Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    bf16=torch.cuda.is_available(),  # ✅ Use bf16 if available
    fp16=False,  # ❌ Disable fp16 if bf16 is available
    save_safetensors=True,
    gradient_checkpointing=True,
    report_to="none"
)

# ✅ Check for existing checkpoint
checkpoint_path = "./results/checkpoint-last"
resume = os.path.exists(checkpoint_path)

# ✅ Fix compute_loss function
def compute_loss(model, inputs, return_outputs=False, **kwargs):
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move inputs to device
    outputs = model(**inputs)
    loss = outputs.loss.float()  # Convert loss to float32 for gradient tracking
    print(f"⚠ Loss: {loss}, requires_grad: {loss.requires_grad}")
    return (loss, outputs) if return_outputs else loss

# ✅ Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.compute_loss = compute_loss  # Inject fixed function

print("🚀 Starting fine-tuning...")

# ✅ Start Training (Auto-resume if checkpoint exists)
trainer.train(resume_from_checkpoint=checkpoint_path if resume else None)

print("✅ Fine-tuning completed.")

# ✅ Save Fine-Tuned Model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("📁 Fine-tuned model saved successfully.")