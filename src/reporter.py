import csv

def generate_reports(classifications, questions, output_md, output_csv):
    """
    Generates Markdown and CSV reports based on classifications.
    classifications: [{'id': 1, 'category': 'Math'}]
    questions: [{'id': 1, 'snippet': 'Find the...', ...}]
    """
    print(f"[+] Generating reports: {output_md}, {output_csv}")
    
    # Map snippets for CSV
    question_snippets = {q['id']: q['snippet'] for q in questions}
    
    # 1. Generate CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Question Number", "Category", "Snippet"])
        for c in classifications:
            # Robust key retrieval
            q_id = c.get('id', c.get('Id', c.get('q_num', 0)))
            raw_cat = c.get('category') or c.get('Category') or 'Unclassified'
            q_cat = str(raw_cat).strip()
            writer.writerow([q_id, q_cat, question_snippets.get(q_id, '')])
            
    # 2. Group for Markdown
    category_map = {}
    for c in classifications:
        q_id = c.get('id', c.get('Id', c.get('q_num', 0)))
        raw_cat = c.get('category') or c.get('Category') or 'Unclassified'
        q_cat = str(raw_cat).strip()
        if q_cat not in category_map:
            category_map[q_cat] = []
        category_map[q_cat].append(q_id)
        
    # Generate Markdown
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# Classified Questions Summary\n\n")
        
        for cat in sorted(category_map.keys()):
            f.write(f"## {cat}\n")
            # Sort the question numbers
            q_nums = sorted(category_map[cat])
            f.write(f"{', '.join(map(str, q_nums))}\n\n")
            
    print("[+] Reports generated successfully.")
