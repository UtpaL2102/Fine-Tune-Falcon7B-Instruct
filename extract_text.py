import os
import pdfplumber

# Define the folder path where PDFs are stored
pdf_folder = "/content/Selected_20_PDFs"

# Ensure the folder exists
if not os.path.exists(pdf_folder):
    raise FileNotFoundError(f"❌ Folder not found: {pdf_folder}")

# Detect all PDFs inside the folder
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# Extract text from PDFs
def extract_text_from_pdfs(pdf_files):
    text_data = []
    for pdf_file in pdf_files:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                text_data.append(text)
        except Exception as e:
            print(f"❌ Error reading {pdf_file}: {e}")
    return text_data

pdf_texts = extract_text_from_pdfs(pdf_files)

# Save extracted text
output_file = "/content/extracted_text.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for i, text in enumerate(pdf_texts):
        f.write(f"--- PDF {i+1}: {pdf_files[i]} ---\n")
        f.write(text + "\n\n")

print(f"✅ Extraction complete! Saved to {output_file}")
