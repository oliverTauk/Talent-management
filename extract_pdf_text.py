import pypdf
import os

def extract_text_from_pdf(pdf_path, output_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Extracted text to {output_path}")
    except Exception as e:
        print(f"Error extracting from {pdf_path}: {e}")

base_path = r"c:\Users\WIN 11\OneDrive\Desktop\zzzz\Talent-management\Data\Check-ins 2024 2025\2025"
manager_pdf = os.path.join(base_path, "ITG Employee Performance Check-In (Manager questions) - 2025.pdf")
employee_pdf = os.path.join(base_path, "ITG Performance Check-In (Employee questions) - 2025.pdf")

extract_text_from_pdf(manager_pdf, "manager_questions.txt")
extract_text_from_pdf(employee_pdf, "employee_questions.txt")
