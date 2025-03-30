from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model_name = "tiiuae/falcon-7b-instruct"
adapter_path = "/content/drive/MyDrive/dummy_extracted_pdfs/fine_tuned_model"

# ✅ Enable 4-bit Quantization (to avoid memory crash)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
) if torch.cuda.is_available() else None

# ✅ Load base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"  # Automatically place on GPU if available
)

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# ✅ Apply LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)

# ✅ Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ✅ Test it
text = "Answer strictly based on the case file: What was the court's decision regarding Obergefell in this case?"
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
