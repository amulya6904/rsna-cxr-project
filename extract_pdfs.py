import pypdf
import os

def extract_text(pdf_path, txt_path):
    try:
        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Extracted {pdf_path} to {txt_path}")
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")

extract_text('RSNA_CXR_PRD.pdf', 'RSNA_CXR_PRD.txt')
extract_text('Nuvexa_Design_System_v2.pdf', 'Nuvexa_Design_System_v2.txt')
