import os
import argparse
from dotenv import load_dotenv

from src.extractor import extract_text_from_pdf
from src.parser import extract_questions_via_llm
from src.classifier import classify_all
from src.reporter import generate_reports
from src.logger import setup_logger

def main():
    # Initialize unbuffered logger to overwrite run.log
    sys_logger = setup_logger("run.log")
    try:
        # Load environment variables from .env
        load_dotenv()
        
        parser = argparse.ArgumentParser(description="LLM-Based Question Classifier")
        parser.add_argument("--pdf", required=True, help="Path to the PDF file to process.")
        parser.add_argument("--categories", default="category.csv", help="Path to the categories CSV file.")
        parser.add_argument("--output-md", default="output_classified.md", help="Output Markdown file name.")
        parser.add_argument("--output-csv", default="output_classified.csv", help="Output CSV file name.")
        parser.add_argument("--extraction-model", default="gemini-3.1-flash-lite", help="Gemini model for question extraction.")
        parser.add_argument("--classification-model", default="gemini-3.1-flash-lite", help="Gemini model for question classification.")
        parser.add_argument("--chunk-size", type=int, default=5, help="Number of pages per extraction chunk.")
        parser.add_argument("--batch-size", type=int, default=25, help="Number of questions per classification batch.")
        parser.add_argument("--max-questions", type=int, default=150, help="Expected number of questions in the paper.")
        
        args = parser.parse_args()
        
        if not os.path.exists(args.pdf):
            print(f"[-] Error: PDF file {args.pdf} not found.")
            return
            
        if not os.path.exists(args.categories):
            print(f"[-] Error: Categories file {args.categories} not found.")
            return

        # 1. Extraction
        print("\n--- STEP 1: EXTRACTION ---")
        raw_text = extract_text_from_pdf(args.pdf, cache_dir="cache")
        
        # 2. Parsing
        print("\n--- STEP 2: PARSING ---")
        questions = extract_questions_via_llm(raw_text, max_q=args.max_questions, model_name=args.extraction_model, chunk_size=args.chunk_size)
        print(f"[+] Parsed {len(questions)} potential questions.")
        
        if not questions:
            print("[-] No questions were parsed. Exiting.")
            return
            
        # 3. Classification
        print("\n--- STEP 3: CLASSIFICATION ---")
        classifications = classify_all(questions, args.categories, model_name=args.classification_model, batch_size=args.batch_size)
        
        if not classifications:
            print("[-] Classification failed or returned no results.")
            return
            
        print(f"[+] Successfully classified {len(classifications)} questions.")
        
        # 4. Reporting
        print("\n--- STEP 4: REPORTING ---")
        generate_reports(classifications, questions, args.output_md, args.output_csv)
        
        print("\n[+] All steps completed successfully.")
    finally:
        sys_logger.close()

if __name__ == "__main__":
    main()
