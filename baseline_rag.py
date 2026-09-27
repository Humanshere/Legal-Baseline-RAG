import json
import os
from typing import List, Dict, Any

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
from tqdm import tqdm

def setup_corpus(dataset_name="joyboseroy/inIRAC", num_records=45, corpus_ids_file="corpus_ids.json"):
    """Loads the dataset, sorts by citation, and creates the working corpus."""
    print(f"Loading dataset: {dataset_name}")
    try:
        ds = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"Falling back to default loading due to: {e}")
        ds = load_dataset(dataset_name)
        if isinstance(ds, dict):
            first_split = list(ds.keys())[0]
            ds = ds[first_split]

    print(f"Total records in dataset: {len(ds)}")
    
    # If citation is nested inside a 'case' column (as in inIRAC), extract it
    if "citation" not in ds.column_names and "case" in ds.column_names:
        print("Extracting 'citation' from nested 'case' field...")
        def extract_nested_fields(example):
            case_data = example.get("case", {})
            citation_val = case_data.get("citation", "")
            if not citation_val:
                citation_val = case_data.get("name", "")
                
            return {
                "citation": citation_val,
                "case_name": case_data.get("name", ""),
                "court": case_data.get("court", ""),
                "year": case_data.get("year", "")
            }
        ds = ds.map(extract_nested_fields)

    citation_field = "citation"
    if citation_field not in ds.column_names:
        print(f"Warning: 'citation' field not found. Available fields: {ds.column_names}")
        fallback_fields = ['id', 'case_id', 'filename', 'name', 'file', 'case_name']
        for f in fallback_fields:
            if f in ds.column_names:
                citation_field = f
                print(f"Using '{f}' as the identifier field instead of 'citation'.")
                break
        if citation_field not in ds.column_names:
            raise ValueError(f"Dataset does not contain 'citation' or any known identifier field! Columns found: {ds.column_names}")
        
    print(f"Sorting records by '{citation_field}' (alphabetically, case-sensitive)...")
    # sort by identifier alphabetically
    sorted_ds = ds.sort(citation_field)
    
    # Take the first 'num_records'
    working_corpus = sorted_ds.select(range(min(num_records, len(sorted_ds))))
    
    # Save exact citations to corpus_ids.json for reproducibility
    citations = [str(c) for c in working_corpus[citation_field]]
    with open(corpus_ids_file, "w") as f:
        json.dump(citations, f, indent=4)
        
    print(f"Saved {len(citations)} case citations to {corpus_ids_file}")
    
    # Let's map it to "citation" column name so downstream code expecting "citation" still works
    if citation_field != "citation":
        # Check if we can rename
        if "citation" not in working_corpus.column_names:
            working_corpus = working_corpus.rename_column(citation_field, "citation")
    
    return working_corpus

def chunk_text(working_corpus, embed_model_name, chunk_size=500, chunk_overlap=50) -> List[Dict[str, Any]]:
    """Chunks the raw judgment text field into overlapping chunks. 
       If no single raw text field exists, concatenates available IRAC fields 
       to form a plain-text baseline without structural annotations."""
    print(f"Chunking documents (~{chunk_size} tokens, {chunk_overlap} overlap)...")
    
    # Use the model's exact tokenizer for precise token chunking
    tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = []
    
    for row in tqdm(working_corpus, desc="Chunking records"):
        # If there's a dedicated raw text field, use it. Otherwise, concatenate IRAC text fields.
        text = ""
        if "text" in row and isinstance(row["text"], str) and row["text"].strip():
            text = row["text"]
        elif "judgment" in row and isinstance(row["judgment"], str) and row["judgment"].strip():
            text = row["judgment"]
        else:
            # Flatten IRAC fields into plain text, discarding structure for the baseline
            parts = []
            
            # Extract issues
            if "issues" in row and isinstance(row["issues"], list):
                for issue in row["issues"]:
                    if isinstance(issue, dict) and "text" in issue:
                        parts.append(issue["text"])
                    elif isinstance(issue, str):
                        parts.append(issue)
                        
            # Extract rules
            if "rules" in row and isinstance(row["rules"], list):
                for rule in row["rules"]:
                    if isinstance(rule, dict) and "text" in rule:
                        parts.append(rule["text"])
                    elif isinstance(rule, str):
                        parts.append(rule)
                        
            # Extract analysis_summary
            if "analysis_summary" in row and isinstance(row["analysis_summary"], str):
                parts.append(row["analysis_summary"])
                
            # Extract conclusion
            if "conclusion" in row and isinstance(row["conclusion"], str):
                parts.append(row["conclusion"])
                
            text = "\n\n".join([p for p in parts if p.strip()])
            
        # If absolutely no text was found in IRAC fields, fallback to extraction_notes or dump row
        if not text:
            if "extraction_notes" in row and isinstance(row["extraction_notes"], str) and row["extraction_notes"].strip():
                text = row["extraction_notes"]
            else:
                # Last resort: just stringify the row so we have SOMETHING to index
                text = str(row)
                
        if not text:
             continue
             
        # Extract metadata
        citation = row.get("citation", "Unknown")
        court = row.get("court", "Unknown")
        year = row.get("year", "Unknown")
        
        # Split text
        record_chunks = text_splitter.split_text(text)
        
        for i, chunk_text in enumerate(record_chunks):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "citation": citation,
                    "court": court,
                    "year": year,
                    "chunk_index": i
                }
            })
            
    print(f"Created {len(chunks)} chunks in total.")
    return chunks
    
def embed_and_store(chunks, model_name):
    """Embeds the chunks and stores them in a FAISS Flat L2 index."""
    if not chunks:
        raise ValueError("No chunks were created from the corpus! The documents might be entirely empty.")
        
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Embedding chunks...")
    texts = [chunk["text"] for chunk in chunks]
    # encode outputs a numpy array
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print("Building FAISS index (flat L2, no approximation)...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return index, model, chunks

def query_pipeline(query: str, index, embed_model, chunks, top_k=8):
    """Embeds a query, retrieves top_k chunks, and generates an answer using local Mistral."""
    # Embed query
    query_emb = embed_model.encode([query]).astype('float32')
    
    # Retrieve top-k chunks
    distances, indices = index.search(query_emb, top_k)
    
    retrieved_chunks = []
    retrieved_chunk_ids = []
    
    for rank, idx in enumerate(indices[0]):
        if idx != -1: # Valid index check
            retrieved_chunks.append(chunks[idx])
            retrieved_chunk_ids.append(int(idx))
            
    # Prepare context for the LLM
    context_str = ""
    for i, c in enumerate(retrieved_chunks):
        meta = c["metadata"]
        context_str += f"--- Context Chunk {i+1} | Citation: {meta.get('citation')} ---\n"
        context_str += c["text"].strip() + "\n\n"
        
    system_prompt = (
        "You are a legal research assistant. Answer the user's question using "
        "ONLY the provided context chunks. For every factual or legal claim, "
        "cite the specific case (by citation string) it came from. If the "
        "provided context does not contain enough information to answer, say so "
        "explicitly rather than guessing."
    )
    
    user_prompt = f"Context:\n{context_str}\nQuestion: {query}"
    
    print(f"Sending query to Mistral-7B-Instruct via Ollama...")
    try:
        response = ollama.chat(
            model='mistral',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        )
        answer = response['message']['content']
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        answer = f"ERROR: Failed to generate answer using Ollama. Ensure 'ollama serve' is running and 'mistral' is pulled. Exception: {e}"
    
    # Construct full trace for logging
    trace = {
        "query": query,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_chunks": retrieved_chunks,
        "generated_answer": answer
    }
    
    return trace

def main():
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # 1. Setup dataset and get top 45 by citation
    corpus = setup_corpus()
    
    # 2. Chunk text
    chunks = chunk_text(corpus, embed_model_name)
    
    # 3. Embed chunks and store in FAISS index
    index, embed_model, chunk_data = embed_and_store(chunks, embed_model_name)
    
    # Example queries to test the pipeline
    queries = [
        "What are the considerations for establishing medical negligence?",
        "Can a confession to a police officer be used as evidence?",
        "Under what conditions is specific performance of a contract granted?"
    ]
    
    print("\nStarting evaluation of queries...")
    traces = []
    for q in queries:
        print(f"\n[Query]: {q}")
        trace = query_pipeline(q, index, embed_model, chunk_data)
        traces.append(trace)
        print(f"[Answer]:\n{trace['generated_answer']}\n")
        
    print("Writing traces to query_traces.jsonl...")
    with open("query_traces.jsonl", "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
            
    print("Pipeline complete! Baseline reproducible results saved.")

if __name__ == "__main__":
    main()
