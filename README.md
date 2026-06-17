# RAG PDF Chat Suite

This repository contains three PDF chat/RAG demo scripts using local PDF data and Ollama models.

## Projects included

- `BM25_rag.py` — BM25 keyword-based retrieval (no vector embeddings)
- `Dense_rag.py` — dense retrieval using embeddings + FAISS
- `hybrid_rag.py` — hybrid retrieval combining BM25 + FAISS with Reciprocal Rank Fusion

All scripts use the same sample document:
- `Daivashala_Dhepale_Resume.pdf`

## Requirements

Install dependencies from `requirements.txt`:

```powershell
D:\Rag\rag\venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you are already in the virtual environment, use:

```powershell
python -m pip install -r requirements.txt
```

## Scripts and Usage

### 1) BM25 keyword retrieval

```powershell
python BM25_rag.py
```

What it does:
- Loads the PDF using `pypdf`
- Splits text into overlapping chunks
- Builds a BM25 index with `rank_bm25`
- Retrieves top chunks for each query
- Sends retrieved context to `OllamaLLM` for answer generation

### 2) Dense retrieval with embeddings + FAISS

```powershell
python Dense_rag.py
```

What it does:
- Loads the PDF using `PyPDFLoader`
- Splits text into chunks with `RecursiveCharacterTextSplitter`
- Embeds chunks using `OllamaEmbeddings`
- Creates a FAISS vector store
- Builds a simple QA chain with `OllamaLLM`
- Generates responses from retrieved top results

### 3) Hybrid BM25 + FAISS retrieval

```powershell
python hybrid_rag.py
```

What it does:
- Loads and cleans PDF text
- Chunkifies the document
- Builds a BM25 index for keyword matching
- Builds a FAISS index for semantic similarity
- Retrieves candidates from both systems
- Fuses ranks with Reciprocal Rank Fusion (RRF)
- Uses `OllamaLLM` to generate the final answer from fused top chunks

## Recommended workflow

1. Activate the correct venv for this folder.
2. Install requirements.
3. Run one of the scripts above.
4. Type questions at the prompt.
5. Enter `exit` to quit.

## Notes

- These examples expect a local Ollama installation and access to the configured Ollama models.
- `BM25_rag.py` does not use vector embeddings; it is purely keyword search.
- `Dense_rag.py` and `hybrid_rag.py` require `langchain_community`, `langchain_text_splitters`, `langchain_core`, and `FAISS`.
- `hybrid_rag.py` uses a fusion strategy to combine keyword and semantic results for more robust document retrieval.

## Customization

- Change `pdf_path` in any script to use a different PDF.
- Adjust `chunk_size`, `chunk_overlap`, and `top_k` values to tune retrieval behavior.
- Replace the Ollama model name if you want to use a different local model.
