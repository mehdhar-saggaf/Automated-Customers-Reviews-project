# === app/utils/pdf_utils.py ===
from fpdf import FPDF
import os

# Directory to save generated PDFs
SUMMARY_DIR = "summaries"
os.makedirs(SUMMARY_DIR, exist_ok=True)

def safe_text(text):
    """
    Remove or ignore characters that can't be encoded with latin-1 (used by default in fpdf).
    """
    return text.encode("latin-1", "ignore").decode("latin-1")

def generate_summary_pdf(title: str, summary: str, filename: str) -> str:
    """
    Generate a simple PDF summary with safe encoding (no Unicode font needed).
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Clean text to avoid Unicode errors
    safe_title = safe_text(title)
    safe_summary = safe_text(summary)

    # Write content to PDF
    pdf.multi_cell(0, 10, f"Summary for: {safe_title}\n\n", align='L')
    pdf.multi_cell(0, 10, safe_summary, align='L')

    # Save to file
    file_path = os.path.join(SUMMARY_DIR, f"{filename}.pdf")
    pdf.output(file_path)

    return file_path
