from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Load tokenizer
model_name = "NousResearch/Llama-2-7B-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad token if missing
tokenizer.pad_token = tokenizer.eos_token

# Read extracted text
file_path = "/content/extracted_text.txt"
with open(file_path, "r", encoding="utf-8") as f:
    pdf_texts = [line.strip() for line in f.readlines() if line.strip()]

# Create dataset
dataset = Dataset.from_dict({"text": pdf_texts})

# Tokenization function
def tokenize_function(example):
    encodings = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Save dataset
dataset_path = "/content/tokenized_dataset"
tokenized_dataset.save_to_disk(dataset_path)
print(f"✅ Tokenized dataset saved to {dataset_path}")
