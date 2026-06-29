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
        progress_callback("Reading PDF page count...")
    
    # Set poppler path for Apple Silicon Macs
    poppler_path = "/opt/homebrew/bin" if os.path.exists("/opt/homebrew/bin/pdftoppm") else None
    
    try:
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(pdf_path, poppler_path=poppler_path)
        total_pages = info.get("Pages", 0)
    except Exception as e:
        print(f"[-] Error reading PDF details: {e}")
        raise

    print(f"[+] Found {total_pages} pages. Converting and running OCR page-by-page...")
    
    full_text = []
    for i in range(1, total_pages + 1):
        msg = f"Converting and running OCR: page {i}/{total_pages}..."
        if progress_callback:
            progress_callback(msg)
        print(f"    {msg}")
        
        try:
            # Convert ONLY one page at a time to prevent RAM spikes (OOM crashes) on cloud servers
            page_images = convert_from_path(
                pdf_path, 
                dpi=150, 
                first_page=i, 
                last_page=i, 
                poppler_path=poppler_path
            )
            if page_images:
                page_image = page_images[0]
                page_text = pytesseract.image_to_string(page_image, config='--psm 3')
                full_text.append(f"--- PAGE {i} ---\n{page_text}")
                # Free memory immediately
                page_image.close()
                del page_image
                del page_images
        except Exception as page_err:
            print(f"[-] Error processing page {i}: {page_err}")
            full_text.append(f"--- PAGE {i} ---\n[OCR Error: Could not read page {i}]")
    
    combined_text = "\n\n".join(full_text)
    
    # Save to cache
    with open(cached_txt_path, "w", encoding="utf-8") as f:
        f.write(combined_text)
        
    print(f"[+] Success! Output cached to: {cached_txt_path}")
    return combined_text
