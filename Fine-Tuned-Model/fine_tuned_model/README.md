# Fine-Tuned Falcon-7B-Instruct

This repository contains a fine-tuned version of **Falcon-7B Instruct**, trained on extracted text from 20 PDFs. The purpose of this fine-tuning was to enhance the model's ability to answer questions **strictly based on the provided dataset**, without relying on external knowledge. The fine-tuning process used **LoRA (Low-Rank Adaptation)** to efficiently train the model while maintaining performance.

## Model Details

### Model Description

This model is based on **Falcon-7B Instruct**, an instruction-tuned variant of the Falcon-7B model. The fine-tuning process aimed to align the model’s responses with domain-specific knowledge extracted from PDFs, ensuring better contextual understanding and accuracy in responses.

- **Developed by:** Priyadarshi Utpal  
- **Funded by:** Self-Initiated  
- **Shared by:** [Priyadarshi Utpal](https://github.com/UtpaL2102)  
- **Model type:** Transformer-based Language Model  
- **Language(s):** English  
- **License:** Apache 2.0  
- **Fine-tuned from model:** `tiiuae/falcon-7b-instruct`  

### Model Sources

- **Repository:** [Fine-Tune-Falcon7B-Instruct](https://github.com/UtpaL2102/Fine-Tune-Falcon7B-Instruct)  
- **Original Falcon-7B Instruct Model:** [tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)  

## Uses

### Direct Use

The fine-tuned model is designed for **domain-specific Q&A** tasks based on the extracted knowledge from PDFs. It can be used for:  
- Answering questions directly from the fine-tuned dataset.  
- Enhancing chatbot applications that require contextual knowledge from PDFs.  

### Downstream Use

- Can be adapted for **other domain-specific tasks** such as legal document analysis, financial reports, or research papers.  
- Could be extended with further fine-tuning for **customer support** or **automated document summarization**.  

### Out-of-Scope Use

- The model is **not intended** for open-ended general knowledge queries.  
- It should **not be used for real-time critical decision-making** without human verification.  

## Bias, Risks, and Limitations

### Risks & Bias

- Since the model is fine-tuned on a specific dataset, it might **overfit** to that domain.  
- Any **biases in the original PDFs** will be reflected in the model’s responses.  
- **Hallucination risk** is reduced but not eliminated.  

### Recommendations

Users should verify responses before using them in critical applications. Further fine-tuning or prompt engineering might be needed to adapt the model to new datasets.  

## How to Get Started with the Model

To use the fine-tuned model, follow these steps:

1. **Load the fine-tuned model**  
   The fine-tuned model is stored in the directory:

2. **Install dependencies**  
   pip install torch transformers peft accelerate bitsandbytes pgsql


3. **Load the model in Python**  
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from peft import PeftModel

   base_model = "tiiuae/falcon-7b-instruct"
   tokenizer = AutoTokenizer.from_pretrained(base_model)

   model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", load_in_8bit=True)
   model = PeftModel.from_pretrained(model, "path_to_fine_tuned_model")


4. **Run inference**

   prompt = "What does the dataset contain?"
   inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
   outputs = model.generate(**inputs)
   print(tokenizer.decode(outputs[0]))



## Training Details

### Training Data

The model was fine-tuned on **20 PDF files**, which were first processed using the following pipeline:

1. **Text Extraction**  
   - Used `pdfplumber` to extract text from PDFs.  
   - Stored extracted text in structured text files.  

2. **Tokenization**  
   - Used **LLaMA-2 Tokenizer** (`NousResearch/Llama-2-7B-chat-hf`) to preprocess text.  
   - Converted text into tokenized sequences for training.  

3. **Dataset Preparation**  
   - Structured the dataset into **instruction-response pairs**.  
   - Saved the final dataset in a Hugging Face-compatible format.  

### Training Procedure

- **Base Model Used:** `tiiuae/falcon-7b-instruct`  
- **Fine-Tuning Approach:** LoRA (Low-Rank Adaptation)  
- **Precision:** FP16 (Mixed)  
- **Optimizer:** AdamW  
- **Batch Size:** 4  
- **Gradient Accumulation:** 8 steps  
- **Checkpoint Saving:** Every 50 steps  
- **Hardware Used:** GPU (Colab Pro)  

#### Why LoRA?

LoRA was used for fine-tuning instead of full model fine-tuning because:  
- It **reduces VRAM requirements** significantly.  
- It allows for **efficient training on consumer GPUs**.  
- It enables **modular adapter-based fine-tuning**, where we can load different adapters without modifying the base model.  

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data  
The fine-tuned model was evaluated by querying it with domain-specific questions based on the PDFs.  

#### Evaluation Metrics  
- **Perplexity Score (PPL):** Used to measure how well the model predicts unseen text.  
- **Response Accuracy:** Compared generated responses with ground truth from PDFs.  

### Results  
- The model successfully answered **domain-specific questions** with **high accuracy**.  
- **Reduced hallucination** compared to the base Falcon-7B Instruct.  
- **Faster inference** with **4-bit quantization**.  

## Environmental Impact

- **Hardware Used:** Google Colab Pro GPU  
- **Training Time:** ~3 hours  
- **Compute Region:** USA  
- **Estimated Carbon Emissions:** Minimal, as LoRA significantly reduces training requirements.  

## Technical Specifications

### Model Architecture  
- **Falcon-7B:** A dense transformer model with **7 billion parameters**.  
- **Instruction-Tuned:** Pre-trained for chat-like responses.  

### Compute Infrastructure  
- **Hardware:** NVIDIA Tesla T4 (Colab Pro)  
- **Software:** PyTorch, Hugging Face Transformers, PEFT  

## Citation

If you use this fine-tuned model, please cite:


## Contact

For questions or collaborations, reach out to:  
- **GitHub:** [UtpaL2102](https://github.com/UtpaL2102)  
- **Email:** [Add your email if you want]  

---

This README is now fully formatted and ready to paste into your GitHub repository! 🚀


