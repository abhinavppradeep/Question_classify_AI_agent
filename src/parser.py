import os
import json
import time
import re
import google.generativeai as genai
from pydantic import BaseModel
from typing import List

class ExtractedQuestion(BaseModel):
    id: int
    text: str

class ExtractionResult(BaseModel):
    questions: List[ExtractedQuestion]

def extract_questions_via_llm(text, max_q=150, model_name="gemini-3.1-flash-lite", chunk_size=5, progress_callback=None):
    """
    Chunks raw OCR text page-by-page and uses Gemini to extract clean questions,
    filtering out ad pages, detailed solutions, and headers automatically.
    """
    # Split text into pages
    pages = text.split("--- PAGE ")
    cleaned_pages = []
    for p in pages:
        p_stripped = p.strip()
        if not p_stripped:
            continue
        cleaned_pages.append(p_stripped)
        
    print(f"[+] Found {len(cleaned_pages)} pages to parse.")
    if progress_callback:
        progress_callback(f"Found {len(cleaned_pages)} pages. Preparing chunks...")
        
    # Group pages into chunks dynamically based on chunk_size parameter
    page_chunks = [cleaned_pages[i:i + chunk_size] for i in range(0, len(cleaned_pages), chunk_size)]
    
    # Setup Gemini API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=api_key)
    
    # Format model name
    full_model_name = model_name if model_name.startswith("models/") else f"models/{model_name}"
    model = genai.GenerativeModel(full_model_name)
    
    all_extracted_questions = []
    
    for idx, chunk in enumerate(page_chunks):
        msg = f"Parsing page chunk {idx+1}/{len(page_chunks)}..."
        print(f"[+] {msg}")
        if progress_callback:
            progress_callback(msg)
            
        chunk_text = "\n\n".join(chunk)
        
        prompt = f"""
        You are an expert OCR parser. Your job is to extract all exam questions from the following raw text.
        
        Rules:
        1. Extract the actual question number (as id) and the full text of the question (including its options (a), (b), (c), (d) if present).
        2. Strictly ignore any page headers, footers, exam dates (e.g. '08-02-2026', 'EASU 08-02-2026'), cover sheets, expected cutoffs, ad pages, or detailed explanations/solutions (ignore text starting with 'End of Solution' or similar solution text).
        3. Do NOT extract solutions. Only extract the question itself.
        4. If a question is split across chunk boundaries, extract the part present in this chunk.
        
        Raw Text:
        {chunk_text}
        """
        
        retries = 3
        while retries > 0:
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=ExtractionResult,
                    )
                )
                result = json.loads(response.text)
                if result and 'questions' in result:
                    all_extracted_questions.extend(result['questions'])
                break
            except Exception as e:
                retries -= 1
                print(f"[-] Error parsing chunk: {e}. Retrying in 10s... ({retries} retries left)")
                time.sleep(10)
                if retries == 0:
                    print("[-] Failed to parse chunk after multiple retries.")
                    
        # Sleep to respect 15 RPM rate limit (60s / 15 = 4s sleep)
        time.sleep(5)
        
    # Align and format results to guarantee exactly 150 questions
    final_questions = [None] * (max_q + 1)
    unplaced_questions = []
    
    for q in all_extracted_questions:
        # Pydantic or dict check
        q_id = q.get('id') if isinstance(q, dict) else getattr(q, 'id', None)
        q_text = q.get('text', '').strip() if isinstance(q, dict) else getattr(q, 'text', '').strip()
        
        if not q_text:
            continue
        if q_id and 1 <= q_id <= max_q:
            if final_questions[q_id] is None:
                final_questions[q_id] = q_text
            else:
                unplaced_questions.append(q_text)
        else:
            unplaced_questions.append(q_text)
            
    # Fill in empty spots sequentially
    unplaced_idx = 0
    for qn in range(1, max_q + 1):
        if final_questions[qn] is None:
            if unplaced_idx < len(unplaced_questions):
                final_questions[qn] = unplaced_questions[unplaced_idx]
                unplaced_idx += 1
            else:
                final_questions[qn] = f"Question {qn} text missing from OCR."
                
    results = []
    for qn in range(1, max_q + 1):
        text = final_questions[qn]
        snippet = text[:150]
        results.append({
            'id': qn,
            'text': text,
            'snippet': snippet
        })
        
    return results
