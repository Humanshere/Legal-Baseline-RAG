import json
import os
from typing import List, Dict, Any

from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    
    if "citation" not in ds.column_names:
        raise ValueError("Dataset does not contain a 'citation' field!")
        
    print("Sorting records by 'citation' (alphabetically, case-sensitive)...")
    # sort by citation alphabetically
    sorted_ds = ds.sort("citation")
    
    # Take the first 'num_records'
    working_corpus = sorted_ds.select(range(min(num_records, len(sorted_ds))))
    
    # Save exact citations to corpus_ids.json for reproducibility
    citations = working_corpus["citation"]
    with open(corpus_ids_file, "w") as f:
        json.dump(citations, f, indent=4)
        
    print(f"Saved {len(citations)} case citations to {corpus_ids_file}")
    return working_corpus

def chunk_text(working_corpus, embed_model_name, chunk_size=500, chunk_overlap=50) -> List[Dict[str, Any]]:
    """Chunks the raw judgment text field into overlapping chunks."""
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
    
    # Identify the raw text field dynamically, bypassing pre-extracted IRAC fields
    features = working_corpus.column_names
    raw_text_field = 'text'
    if 'text' not in features:
        if 'judgment' in features:
            raw_text_field = 'judgment'
        elif 'judgment_text' in features:
            raw_text_field = 'judgment_text'
        else:
            # Fallback search for a plausible text column
            for col in features:
                if 'text' in col or 'judg' in col:
                    raw_text_field = col
                    break
    
    print(f"Using '{raw_text_field}' field for raw judgment text.")
    
    for row in tqdm(working_corpus, desc="Chunking records"):
        text = row.get(raw_text_field, "")
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
