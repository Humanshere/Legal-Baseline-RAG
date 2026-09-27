# Indian Legal QA - Baseline RAG Pipeline

This repository contains a simple, unmodified baseline Retrieval-Augmented Generation (RAG) pipeline for Indian Legal QA. It is intentionally designed as a "no safeguard" baseline for research comparisons, meaning it lacks any verification steps, graph structures, or custom citation-checking logic. It strictly uses standard chunking, embedding, vector retrieval, and local LLM generation.

## Prerequisites

1. **Python 3.8+**
2. **Ollama**: You must have [Ollama](https://ollama.com/) installed and running locally in the background. 
3. **Mistral-7B-Instruct**: Pull the model in Ollama before running the script:
   ```bash
   ollama pull mistral
   ```

## Installation

Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the Pipeline

Execute the main Python script to run the full pipeline:
```bash
python baseline_rag.py
```

### Reproducing the 45-Case Corpus and Re-running Queries
When you run `python baseline_rag.py`, the script automatically reproduces the exact conditions required for this baseline:

1. **Dataset Loading & Corpus Creation**: 
   - Downloads the `joyboseroy/inIRAC` dataset from HuggingFace.
   - Sorts the dataset alphabetically (case-sensitive string sort) by the `citation` field.
   - Extracts the first 45 records to form the precise working corpus.
   - Saves this exact list of 45 case citations to `corpus_ids.json`.
2. **Chunking**: 
   - Uses the raw judgment text field to ensure a fair "no graph/no annotations" comparison.
   - Chunks text into ~500-token chunks with a 50-token overlap using the embedding model's exact HuggingFace tokenizer. 
   - Preserves standard metadata: `citation`, `court`, `year`, and `chunk_index`.
3. **Embedding & Storage**: 
   - Embeds the chunks using `sentence-transformers/all-MiniLM-L6-v2`.
   - Stores the vectors in a flat L2 FAISS index (no approximation) for deterministic, reproducible retrieval.
4. **Query & Generation**:
   - Queries are embedded, and the top-8 most similar chunks are retrieved via cosine similarity.
   - These chunks are passed to `Mistral-7B-Instruct` via the local Ollama client, enforcing a strict system prompt that demands case citations for every claim and prevents guessing.
5. **Trace Logging**:
   - Generates a full trace for every query, saved line-by-line to `query_traces.jsonl`. This log includes the query text, retrieved chunk IDs, raw retrieved texts, generated answers, and citations.

To test your own queries or evaluate the model on an existing benchmark, simply update the `queries` list within the `main()` function of `baseline_rag.py` and run the script again. The new traces will be exported to `query_traces.jsonl`.
