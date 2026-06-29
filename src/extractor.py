import os
from pdf2image import convert_from_path
import pytesseract

if os.path.exists("/opt/homebrew/bin/tesseract"):
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

def extract_text_from_pdf(pdf_path, cache_dir="cache", progress_callback=None):
    """
    Extracts text from a PDF file using OCR (pytesseract).
    Caches the extracted text in a .txt file inside cache_dir.
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    base_name = os.path.basename(pdf_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    cached_txt_path = os.path.join(cache_dir, f"txt_{file_name_without_ext}.txt")
    
    if os.path.exists(cached_txt_path):
        if progress_callback:
            progress_callback(f"Found cached text file for {pdf_path}. Loading...")
        print(f"[+] Found cached text file for {pdf_path}. Loading from {cached_txt_path}...")
        with open(cached_txt_path, 'r', encoding='utf-8') as f:
            return f.read()

    if progress_callback:
        progress_callback("Converting PDF pages to images (this can take a moment)...")
    print(f"[+] Converting {pdf_path} pages to images (this might take a while)...")
    try:
        # Set poppler path for Apple Silicon Macs
        poppler_path = "/opt/homebrew/bin" if os.path.exists("/opt/homebrew/bin/pdftoppm") else None
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    except Exception as e:
        print(f"[-] Error reading PDF file. Ensure 'poppler' is installed. Details: {e}")
        raise

    full_text = []
    total_pages = len(pages)
    
    print(f"[+] Running OCR on {total_pages} page(s)...")
    for i, page in enumerate(pages):
        msg = f"Running OCR: processing page {i + 1}/{total_pages}..."
        if progress_callback:
            progress_callback(msg)
        print(f"    {msg}")
        page_text = pytesseract.image_to_string(page, config='--psm 3')
        full_text.append(f"--- PAGE {i + 1} ---\n{page_text}")
    
    combined_text = "\n\n".join(full_text)
    
    # Save to cache
    with open(cached_txt_path, "w", encoding="utf-8") as f:
        f.write(combined_text)
        
    print(f"[+] Success! Output cached to: {cached_txt_path}")
    return combined_text
