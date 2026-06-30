import os
import json
import time
import google.generativeai as genai
from pydantic import BaseModel
from typing import List

class Classification(BaseModel):
    id: int
    category: str

class BatchClassificationResult(BaseModel):
    classifications: List[Classification]

def setup_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in the environment.")
    genai.configure(api_key=api_key)

def get_batches(questions, batch_size=25):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(questions), batch_size):
        yield questions[i:i + batch_size]

def classify_questions_batch(questions, categories, model_name="gemini-3.1-flash-lite"):
    """
    Classifies a batch of questions using the Gemini API.
    Uses Structured Outputs to enforce JSON format.
    """
    full_model_name = model_name if model_name.startswith("models/") else f"models/{model_name}"
    model = genai.GenerativeModel(full_model_name)
    
    categories_str = ", ".join(categories)
    
    prompt = f"""
    You are an expert question classifier. You will be provided with a batch of numbered questions and a list of valid categories.
    For each question, assign the single most appropriate category from the list. You must pick the closest matching category.
    
    Valid Categories:
    {categories_str}
    
    Do NOT classify any question as 'Unclassified' unless the text is completely missing, blank, or contains a placeholder like "Question X text missing".
    
    Questions:
    {json.dumps([{"id": q["id"], "text": q["text"]} for q in questions], indent=2)}
    """
    
    retries = 3
    delay = 10
    for attempt in range(retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=BatchClassificationResult,
                ),
            )
            # Try parsing the JSON response
            return json.loads(response.text)
        except json.JSONDecodeError as je:
            print(f"[-] JSON Decode Error (Attempt {attempt+1}/{retries}): {je}")
            if attempt == retries - 1:
                return None
            time.sleep(2)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "ResourceExhausted" in err_msg or "Quota" in err_msg:
                print(f"[-] Rate limit hit (429). Retrying in {delay}s... (Attempt {attempt+1}/{retries})")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"[-] API Error (Attempt {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    return None
                time.sleep(2)

def classify_all(questions, categories_file, model_name="gemini-3.1-flash-lite", batch_size=25, progress_callback=None):
    """
    Reads categories, batches questions, and classifies them all.
    """
    setup_gemini()
    
    with open(categories_file, 'r', encoding='utf-8') as f:
        categories = [line.strip() for line in f if line.strip()]
        
    print(f"[+] Loaded {len(categories)} categories.")
    
    all_classifications = []
    batches = list(get_batches(questions, batch_size))
    
    print(f"[+] Processing {len(batches)} batches...")
    
    for i, batch in enumerate(batches):
        msg = f"Classifying batch {i+1}/{len(batches)} (size: {len(batch)})..."
        if progress_callback:
            progress_callback(msg)
        print(f"    {msg}")
        result = classify_questions_batch(batch, categories, model_name=model_name)
        
        if result and 'classifications' in result:
            all_classifications.extend(result['classifications'])
        else:
            warn_msg = f"[-] Failed to classify batch {i+1}. Attempting sub-batch fallback..."
            if progress_callback:
                progress_callback(warn_msg)
            print(f"    {warn_msg}")
            
            # Fallback Level 1: Sub-batches of 5 questions
            sub_batches = list(get_batches(batch, 5))
            for sj, sub_batch in enumerate(sub_batches):
                sub_msg = f"[Fallback] Classifying sub-batch {sj+1}/{len(sub_batches)} (size: {len(sub_batch)})..."
                if progress_callback:
                    progress_callback(sub_msg)
                print(f"        {sub_msg}")
                sub_result = classify_questions_batch(sub_batch, categories, model_name=model_name)
                
                if sub_result and 'classifications' in sub_result:
                    all_classifications.extend(sub_result['classifications'])
                else:
                    # Fallback Level 2: Individual question classification
                    for q in sub_batch:
                        ind_msg = f"[Individual Fallback] Classifying question {q['id']}..."
                        if progress_callback:
                            progress_callback(ind_msg)
                        print(f"            {ind_msg}")
                        single_result = classify_questions_batch([q], categories, model_name=model_name)
                        if single_result and 'classifications' in single_result and len(single_result['classifications']) > 0:
                            all_classifications.extend(single_result['classifications'])
                        else:
                            print(f"            [-] Failed to classify question {q['id']} individually.")
                # Sleep to respect rate limit (15 RPM max) during sub-batching fallback
                time.sleep(6)
            
        # Rate limit protection for free tier (5 RPM limit)
        # Sleeping for 15 seconds guarantees we stay well under the limit
        time.sleep(15)
        
    # Deduplicate results: ensure each question ID appears exactly once and standardize as dict
    deduped = {}
    for c in all_classifications:
        q_id = c.get('id') if isinstance(c, dict) else getattr(c, 'id', None)
        q_cat = c.get('category') if isinstance(c, dict) else getattr(c, 'category', 'Unclassified')
        
        if q_id is not None:
            q_id = int(q_id)
            if q_id in deduped:
                # Prefer real category over Unclassified
                if deduped[q_id]['category'] == 'Unclassified' and q_cat != 'Unclassified':
                    deduped[q_id] = {'id': q_id, 'category': q_cat}
            else:
                deduped[q_id] = {'id': q_id, 'category': q_cat}
                
    # Fill in any missing IDs in classifications up to len(questions) just to be safe
    for q in questions:
        q_id = int(q['id'])
        if q_id not in deduped:
            deduped[q_id] = {'id': q_id, 'category': 'Unclassified'}
            
    # --- SECONDARY CLEANUP PHASE ---
    # Extract all questions that remain Unclassified (and have valid text)
    unclassified_q = []
    for q in questions:
        q_id = int(q['id'])
        if deduped[q_id]['category'] == 'Unclassified':
            q_text = q.get('text', '')
            # If the OCR extraction legitimately failed, do not try to reclassify it.
            if "text missing from OCR" not in q_text:
                unclassified_q.append(q)
            else:
                print(f"    [Cleanup] Skipping Question {q_id} as its text is missing from OCR.")
            
    if unclassified_q:
        cleanup_msg = f"[+] Cleanup Phase: Found {len(unclassified_q)} legitimately unclassified questions. Retrying with small batch size 15..."
        if progress_callback:
            progress_callback(cleanup_msg)
        print(f"\n{cleanup_msg}")
        
        # Split into small batches of 15 questions
        cleanup_batches = list(get_batches(unclassified_q, 15))
        for k, clean_batch in enumerate(cleanup_batches):
            clean_batch_msg = f"    [Cleanup] Classifying batch {k+1}/{len(cleanup_batches)} (size: {len(clean_batch)})..."
            if progress_callback:
                progress_callback(clean_batch_msg)
            print(clean_batch_msg)
            
            clean_result = classify_questions_batch(clean_batch, categories, model_name=model_name)
            
            if clean_result and 'classifications' in clean_result:
                for c in clean_result['classifications']:
                    cq_id = c.get('id') if isinstance(c, dict) else getattr(c, 'id', None)
                    cq_cat = c.get('category') if isinstance(c, dict) else getattr(c, 'category', 'Unclassified')
                    if cq_id is not None:
                        cq_id = int(cq_id)
                        if cq_cat != 'Unclassified':
                            deduped[cq_id] = {'id': cq_id, 'category': cq_cat}
            # Sleep to respect rate limits during cleanup
            time.sleep(6)
            
    return list(deduped.values())
