import os
import json
from datetime import datetime
import fitz  # PyMuPDF
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
import re

# --- CONFIGURATION ---
DEVICE = "cpu"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL_PATH = "./reasoning-llama3.2-1b.Q6_K.gguf"  # Updated model path
TOP_N_TO_SUMMARIZE = 5
CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200
MIN_CHUNK_WORDS = 200

# --- 1. MODEL SETUP ---
def setup_models():
    """Load embedding model and the Llama 3.2 LLM."""
    print(f"Loading models onto {DEVICE}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    llm = None
    if os.path.exists(LLM_MODEL_PATH):
        try:
            # Optimized settings for Llama 3.2 1B reasoning model
            llm = Llama(
                model_path=LLM_MODEL_PATH,
                n_ctx=4096,  # Increased context for better reasoning
                n_gpu_layers=0,
                verbose=False,
                n_threads=4,  # Optimize for CPU
                use_mmap=True,
                use_mlock=False
            )
            print("Llama 3.2 1B Reasoning model loaded successfully.")
        except Exception as e:
            print(f"WARNING: Could not load Llama model. Error: {e}")
    else:
        print(f"WARNING: Llama model not found at {LLM_MODEL_PATH}.")
    return embedding_model, llm

# --- SIMPLE BUT BRUTAL STRICT CHECKER ---
def check_relevance_llama32(text: str, persona: str, job_to_be_done: str, llm=None):
    """
    SIMPLE but BRUTAL elimination for small model with proper response handling.
    """
    if llm is None:
        print("   → Using fallback relevance check (no LLM)")
        return check_relevance_fallback(text, persona, job_to_be_done)

    # Pre-screening for obvious conflicts
    text_lower = text.lower()
    job_lower = job_to_be_done.lower()
    # --- INSTANT ELIMINATION SECTION ---
    elimination_rules = {
        'vegetarian': {
            'keywords': ['vegetarian', 'veg'],
            'banned': [
                'meat', 'chicken', 'beef', 'pork', 'fish', 'lamb', 'turkey', 'bacon', 'ham', 'mutton',
                'egg', 'eggs', 'shrimp', 'crab', 'lobster', 'oyster', 'anchovy', 'gravy', 'salami'
            ]
        },
        'budget': {
            'keywords': ['budget', 'cheap', 'affordable', 'low cost', 'economical'],
            'banned': ['expensive', 'premium', 'luxury', 'costly', 'high-end', 'posh', 'exclusive']
        },
        'kids': {
            'keywords': ['kid', 'child', 'children', 'toddler', 'baby'],
            'banned': ['alcohol', 'vodka', 'rum', 'whiskey', 'bourbon', 'beer', 'wine', 'martini', 'tequila']
        },
        'quick': {
            'keywords': ['quick', 'fast', 'easy', 'instant', 'rapid'],
            'banned': ['slow-cooked', 'overnight', 'long prep', 'requires hours']
        },
        'healthy': {
            'keywords': ['healthy', 'low fat', 'low sugar', 'clean eating', 'nutrition'],
            'banned': ['fried', 'greasy', 'deep fried', 'sugar-laden', 'high sugar', 'unhealthy', 'processed']
        },
        'alcohol-free': {
            'keywords': ['non alcoholic', 'alcohol free', 'sober'],
            'banned': ['vodka', 'rum', 'whiskey', 'gin', 'beer', 'cocktail', 'brandy', 'alcohol']
        },
        'non-tech': {
            'keywords': ['non tech', 'non-technical', 'beginner', 'basic'],
            'banned': ['machine learning', 'API', 'infrastructure', 'DevOps', 'kubernetes', 'docker', 'cloud']
        },
        'vegan': {
            'keywords': ['vegan', 'plant-based'],
            'banned': ['milk', 'cheese', 'butter', 'cream', 'yogurt', 'honey', 'egg', 'meat', 'gelatin']
        },
        'gluten-free': {
            'keywords': ['gluten free', 'celiac'],
            'banned': ['wheat', 'bread', 'pasta', 'barley', 'rye', 'flour', 'gluten']
        }
    }

    for category, rules in elimination_rules.items():
        if any(kw in job_lower for kw in rules['keywords']):
            for bad_word in rules['banned']:
                if bad_word in text_lower:
                    print(f"   → ELIMINATED: Found '{bad_word}' in '{category}'-sensitive task")
                    return False

    # Truncate text for small model
    text_words = text.split()
    if len(text_words) > 150:  # Even shorter for reliability
        text = ' '.join(text_words[:150])
    
    # SIMPLE, DIRECT prompt for small model
    simple_prompt = f"""Task: {persona} needs {job_to_be_done}

Content: "{text}"

Question: Is this content relevant to "{job_to_be_done}" Be very strict if it contains even little bit irrelevant content SAY NO?

Answer (YES or NO):"""

    try:
        print(f"   → Checking relevance with LLM...")
        response = llm(
            simple_prompt,
            max_tokens=3,
            temperature=0.0,
        )
        
        # This part has been simplified as the raw response handling logic was complex
        # and likely specific to a previous version of llama-cpp-python.
        # This now handles the typical dictionary response format.
        if isinstance(response, dict) and 'choices' in response and response['choices']:
            response_text = response['choices'][0]['text'].strip().upper()
        else:
             print(f"   → Unexpected response format: {type(response)}")
             print(f"   → Using fallback relevance check")
             return check_relevance_fallback(text, persona, job_to_be_done)
        
        print(f"   → Model response: '{response_text}'")
        
        # More flexible response parsing
        if "YES" in response_text:
            print("   → KEPT (passed strict check)")
            return True
        elif "NO" in response_text:
            print("   → ELIMINATED")
            return False
        else:
            print(f"   → UNCLEAR RESPONSE '{response_text}' → Using fallback")
            return check_relevance_fallback(text, persona, job_to_be_done)

    except Exception as e:
        print(f"   → ERROR: {e} → Using fallback")
        import traceback
        traceback.print_exc()
        return check_relevance_fallback(text, persona, job_to_be_done)

# This function is no longer called to generate the final output, but is kept to not alter the original script's full contents.
def improved_summarizer_llama32(text: str, persona: str, job_to_be_done: str, llm=None):
    """
    Solution extractor that provides actionable information for the persona to complete their job.
    """
    # This function's logic remains but its output is not used in the new format.
    if llm is None:
        return create_intelligent_fallback_summary(text, persona, job_to_be_done)
    text_words = text.split()
    if len(text_words) > 150:
        text = ' '.join(text_words[:150])
    summary_prompt = f"""I am a {persona} and I need to {job_to_be_done}.
From this content: "{text}"
What specific information helps me {job_to_be_done}? Give me actionable details:
Solution:"""
    try:
        response = llm(summary_prompt, max_tokens=250, temperature=0.2)
        if isinstance(response, dict) and 'choices' in response and response['choices']:
            solution = response['choices'][0]['text'].strip()
            if solution and len(solution.strip()) > 10:
                return solution
    except Exception:
        pass
    return create_intelligent_fallback_summary(text, persona, job_to_be_done)


# --- HELPER FUNCTIONS ---
def parse_pdf_to_chunks(doc_path: str):
    """
    Extracts text from a PDF and splits it into fixed-size, overlapping chunks.
    """
    final_chunks = []
    try:
        doc = fitz.open(doc_path)
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text").strip().replace('\n', ' ')
            page_text = ' '.join(page_text.split())

            if not page_text or len(page_text.split()) < MIN_CHUNK_WORDS:
                continue

            start_index = 0
            while start_index < len(page_text):
                end_index = start_index + CHUNK_SIZE_CHARS
                chunk_text = page_text[start_index:end_index]

                if len(chunk_text.strip().split()) > MIN_CHUNK_WORDS:
                    final_chunks.append({
                        "text": chunk_text.strip(),
                        "page_number": page_num + 1,
                        "source_doc": os.path.basename(doc_path)
                    })
                
                next_start = start_index + CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS
                if next_start <= start_index:
                    break
                start_index = next_start
                
    except Exception as e:
        print(f"WARNING: Could not parse {doc_path}: {e}")
    return final_chunks

def calculate_all_scores(query: str, chunks: list, embedding_model):
    """Calculates similarity scores for all chunks and sorts them."""
    if not chunks: 
        return []
    
    query_embedding = embedding_model.encode([query], convert_to_tensor=True, device=DEVICE)
    chunk_texts = [chunk['text'] for chunk in chunks]
    chunk_embeddings = embedding_model.encode(chunk_texts, convert_to_tensor=True, device=DEVICE)
    similarities = cosine_similarity(query_embedding.cpu(), chunk_embeddings.cpu())[0]
    
    for i, chunk in enumerate(chunks):
        chunk['relevance_score'] = float(similarities[i])
    
    return sorted(chunks, key=lambda x: x['relevance_score'], reverse=True)

def create_intelligent_fallback_summary(text: str, persona: str, job_to_be_done: str):
    """Create a more intelligent fallback summary using keyword extraction and filtering."""
    persona_keywords = extract_keywords(persona.lower())
    job_keywords = extract_keywords(job_to_be_done.lower())
    all_target_keywords = persona_keywords + job_keywords
    sentences = re.split(r'[.!?]+', text)
    relevant_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) < 5:
            continue
        score = sum(1 for keyword in all_target_keywords if keyword in sentence.lower())
        if score > 0:
            relevant_sentences.append((sentence, score))
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    if relevant_sentences:
        top_sentences = [sent[0] for sent in relevant_sentences[:3]]
        summary = '. '.join(top_sentences)
        if not summary.endswith('.'):
            summary += '.'
        return f"Relevant for {persona}: {summary}"
    else:
        words = text.split()
        summary = ' '.join(words[:40]) + "..." if len(words) > 40 else ' '.join(words)
        return f"Content for {persona} consideration: {summary}"

def extract_keywords(text):
    """Extract meaningful keywords from text, removing common words."""
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 
        'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 
        'will', 'with', 'this', 'but', 'they', 'have', 'had', 'what', 'said', 
        'each', 'which', 'she', 'do', 'how', 'their', 'if', 'up', 'out', 'my'
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = [word for word in words if word not in stop_words]
    return list(set(keywords))

def check_relevance_fallback(text: str, persona: str, job_to_be_done: str):
    """ Fallback relevance checking using keyword analysis. """
    all_keywords = extract_keywords(persona.lower()) + extract_keywords(job_to_be_done.lower())
    text_lower = text.lower()
    matches = sum(1 for keyword in all_keywords if len(keyword) >= 3 and keyword in text_lower)
    keyword_threshold = max(1, len(all_keywords) // 4)
    is_relevant = matches >= keyword_threshold
    print(f"   Fallback relevance: {matches}/{len(all_keywords)} keywords matched (threshold: {keyword_threshold}) -> {'RELEVANT' if is_relevant else 'NOT RELEVANT'}")
    return is_relevant

# * NEW HELPER FUNCTION TO MATCH DESIRED OUTPUT *
def generate_section_title(text: str) -> str:
    """Generates a title from the first sentence or line of a text chunk."""
    # Split by the first major punctuation, newline, or bullet point
    parts = re.split(r'[.!?\n•]', text, 1)
    title = parts[0].strip()
    # Clean up extra whitespace
    title = ' '.join(title.split())
    # Cap title length to avoid overly long titles
    if len(title.split()) > 15:
        title = ' '.join(title.split()[:15]) + '...'
    # If the title is extremely short, it might not be descriptive
    if not title or len(title) < 10:
        return "Relevant Document Section"
    return title

# --- MAIN PIPELINE (MODIFIED) ---
def intelligent_document_analyst(doc_paths: list, persona: str, job_to_be_done: str, collection_path: str):
    embedding_model, llm = setup_models()

    print("\n1. Parsing All Documents...")
    all_chunks = []
    for path in doc_paths:
        chunks = parse_pdf_to_chunks(path)
        all_chunks.extend(chunks)
        print(f"   Parsed {len(chunks)} chunks from {os.path.basename(path)}")
    
    if not all_chunks:
        print("ERROR: No text extracted from documents.")
        return None
    print(f"   Found {len(all_chunks)} total text chunks.")

    print("\n2. Calculating Relevance Scores for ALL Chunks...")
    full_query = f"Information for a {persona} trying to {job_to_be_done}."
    all_scored_chunks = calculate_all_scores(full_query, all_chunks, embedding_model)
    print(f"   Scores calculated for all {len(all_scored_chunks)} chunks.")

    # Log top scores for debugging
    print("   Top 5 relevance scores:")
    for i, chunk in enumerate(all_scored_chunks[:5]):
        print(f"   {i+1}. Score: {chunk['relevance_score']:.4f} from {chunk['source_doc']}")

    # The log file part is kept as it is useful for debugging.
    log_path = os.path.join(collection_path, 'all_findings_with_scores.txt')
    print(f"\n3. Storing complete ranked list to: {log_path}")
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Complete ranked list for job: '{job_to_be_done}'\n" + "="*80 + "\n\n")
            for i, chunk in enumerate(all_scored_chunks):
                f.write(f"--- Rank {i+1} | Score: {chunk['relevance_score']:.4f} ---\n")
                f.write(f"Source: {chunk['source_doc']}, Page: {chunk['page_number']}\n\n")
                f.write(f"{chunk['text']}\n" + "="*80 + "\n\n")
        print("   ✓ Log file saved successfully.")
    except Exception as e:
        print(f"   ERROR: Could not write log file: {e}")

    print(f"\n4. Using Simple Strict Filter with Llama 3.2...")
    
    # * MODIFIED: Initialize output dictionary in the desired format *
    output = {
        "metadata": {
            "input_documents": [os.path.basename(p) for p in doc_paths],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }
    
    relevant_chunks_found = 0
    chunks_processed = 0
    max_chunks_to_check = len(all_scored_chunks)
    
    print(f"\n5. Processing chunks with STRICT elimination until {TOP_N_TO_SUMMARIZE} relevant ones found...")
    
    for chunk in all_scored_chunks:
        if relevant_chunks_found >= TOP_N_TO_SUMMARIZE:
            break
            
        chunks_processed += 1
        print(f"\n--- Processing chunk {chunks_processed} (looking for relevant #{relevant_chunks_found + 1}) ---")
        print(f"   Source: {chunk['source_doc']}, Page: {chunk['page_number']}, Score: {chunk['relevance_score']:.4f}")
        
        is_relevant = check_relevance_llama32(chunk['text'], persona, job_to_be_done, llm)

        if is_relevant:
            print(f"   ✓ SURVIVED STRICT CHECK - Formatting output...")
            relevant_chunks_found += 1
            
            # * MODIFIED: Populate the new structure instead of the old one *
            section_title = generate_section_title(chunk['text'])

            # Append to the "extracted_sections" list
            output["extracted_sections"].append({
                "document": chunk['source_doc'],
                "section_title": section_title,
                "importance_rank": relevant_chunks_found,
                "page_number": chunk['page_number']
            })
            
            # Append to the "subsection_analysis" list
            output["subsection_analysis"].append({
                "document": chunk['source_doc'],
                "refined_text": chunk['text'],
                "page_number": chunk['page_number']
            })
            
            print(f"   ✓ Formatted output generated.")
        else:
            print(f"   ✗ ELIMINATED by strict check")
    
    print(f"\n   Final Results: Found {relevant_chunks_found} relevant chunks out of {chunks_processed} processed")
    
    # Handle the fallback case if no chunks are found
    if relevant_chunks_found == 0:
        print("   WARNING: STRICT filter eliminated everything! Using fallback for top 5...")
        for i, chunk in enumerate(all_scored_chunks[:TOP_N_TO_SUMMARIZE]):
            section_title = generate_section_title(chunk['text'])
            
            output["extracted_sections"].append({
                "document": chunk['source_doc'],
                "section_title": section_title,
                "importance_rank": i + 1,
                "page_number": chunk['page_number']
            })
            
            output["subsection_analysis"].append({
                "document": chunk['source_doc'],
                "refined_text": chunk['text'],
                "page_number": chunk['page_number']
            })

    # The subsection_analysis list might not be in rank order, so we sort it
    # based on the order of the extracted_sections to be safe.
    # This step is optional but ensures consistency if the lists get out of sync.
    ranked_docs_pages = [(s["document"], s["page_number"]) for s in output["extracted_sections"]]
    output["subsection_analysis"].sort(key=lambda x: ranked_docs_pages.index((x["document"], x["page_number"])))

    return output


# --- COLLECTION PROCESSING ---
def process_collection(collection_path):
    input_json_path = os.path.join(collection_path, "challenge1b_input.json")
    if not os.path.exists(input_json_path):
        print(f"ERROR: Input JSON not found: {input_json_path}")
        return None
    
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    documents = input_data.get("documents", [])
    persona = input_data.get("persona", {}).get("role", "Professional")
    job_to_be_done = input_data.get("job_to_be_done", {}).get("task", "Complete assigned task")
    
    pdf_dir = os.path.join(collection_path, "PDFs")
    document_filepaths = [os.path.join(pdf_dir, doc.get("filename", "")) for doc in documents]
    document_filepaths = [path for path in document_filepaths if os.path.exists(path)]
    
    if not document_filepaths:
        print(f"ERROR: No valid PDFs found in {pdf_dir}")
        return None
    
    print(f"\n[INFO] Processing {os.path.basename(collection_path)}:")
    print(f"   Persona: {persona}")
    print(f"   Job: {job_to_be_done}")
    print(f"   Documents: {len(document_filepaths)}")
    print(f"   Model: Llama 3.2 1B with STRICT elimination")
    
    return intelligent_document_analyst(document_filepaths, persona, job_to_be_done, collection_path)

if __name__ == "__main__":
    start_time = datetime.now()
    collections_dir = "app"
    collections = [d for d in os.listdir(collections_dir) 
                   if os.path.isdir(os.path.join(collections_dir, d)) and d.startswith("Collection")]
    
    if not collections:
        print("WARNING: No 'Collection *' directories found in the /app directory.")
    
    for collection_name in sorted(collections):
        collection_path = os.path.join(collections_dir, collection_name)
        result = process_collection(collection_path)
        
        if result:
            output_path = os.path.join(collection_path, "challenge1b_output.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"\n✓ SUCCESS: Analysis saved to: {output_path}")
            
            # Print summary stats
            sections_found = len(result.get("extracted_sections", []))
            print(f"   Generated {sections_found} sections.")
        else:
            print(f"✗ ERROR: Failed to process {collection_name}")
        print("-" * 60)
    
    print(f"\nTotal execution time: {datetime.now() - start_time}")