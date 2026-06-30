import os
import json
import time
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List

class Classification(BaseModel):
    id: int = Field(..., ge=1, le=300)
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
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=BatchClassificationResult,
            ),
        )
        try:
            return json.loads(response.text)
        except json.JSONDecodeError as je:
            print(f"[-] JSON Decode Error: {je}")
            print(f"[-] Response length: {len(response.text)}")
            print(f"[-] Response snippet: {response.text[:1000]} ... [TRUNCATED] ... {response.text[-1000:]}")
            return None
    except Exception as e:
        print(f"[-] API Error during classification: {e}")
        return None

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
                # Small sleep to respect rate limit during sub-batching
                time.sleep(2)
            
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
            
    return list(deduped.values())
