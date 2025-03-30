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
