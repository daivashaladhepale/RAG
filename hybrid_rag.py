# HYBRID RAG - BM25 + Dense + RRF Fusion

import re
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate 




# STEP 1: CLEAN EXTRACTED TEXT

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # remove symbols/icons
    text = re.sub(r'\s+', ' ', text)               
    return text.strip()



# STEP 2: LOAD PDF

def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    for page in pages:
        page.page_content = clean_text(page.page_content)  
    print(f" Loaded {len(pages)} pages")
    return pages



# STEP 3: CHUNK

def chunk_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,      
        chunk_overlap=30,
    )
    chunks = splitter.split_documents(pages)
    print(f" Created {len(chunks)} chunks")
    return chunks


# STEP 4: BUILD BM25 INDEX (keyword side)

def tokenize(text):
    return re.findall(r'\b[a-z]+\b', text.lower())

def build_bm25(chunks):
    tokenized = [tokenize(c.page_content) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(" BM25 index built")
    return bm25


# STEP 5: BUILD FAISS INDEX (semantic/vector side)
# 
def build_faiss(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(" FAISS index built")
    return vectorstore


# STEP 6: RRF FUSION

# Combines BM25 ranks + FAISS ranks into one final ranking

def reciprocal_rank_fusion(bm25_results, faiss_results, k=60):
    """
    bm25_results  = list of (chunk_index, score) from BM25
    faiss_results = list of (chunk_index, score) from FAISS
    k = 60 is standard RRF constant (smooths out rank differences)
    """
    rrf_scores = {}

    # Score from BM25 ranks
    for rank, (idx, _) in enumerate(bm25_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + k)

    # Score from FAISS ranks
    for rank, (idx, _) in enumerate(faiss_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + k)

    # Sort by combined RRF score
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_chunks


# STEP 7: RETRIEVE using both BM25 + FAISS then fuse

def hybrid_retrieve(query, bm25, vectorstore, chunks, top_k=3, candidate_k=10):

    print(f"\n Query: '{query}'")
    print("=" * 60)

    # --- BM25 Retrieval ---
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top = sorted(
        enumerate(bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )[:candidate_k]

    print("\n BM25 Top 3:")
    for rank, (idx, score) in enumerate(bm25_top[:3]):
        print(f"  Rank {rank+1} | Score: {score:.4f} | {chunks[idx].page_content[:70]}...")

    # --- FAISS Retrieval ---
    faiss_results_raw = vectorstore.similarity_search_with_score(query, k=candidate_k)
    # Map FAISS results back to chunk indices
    faiss_top = []
    for doc, score in faiss_results_raw:
        for i, chunk in enumerate(chunks):
            if chunk.page_content == doc.page_content:
                faiss_top.append((i, score))
                break

    print("\n FAISS Top 3:")
    for rank, (idx, score) in enumerate(faiss_top[:3]):
        print(f"  Rank {rank+1} | Score: {score:.4f} | {chunks[idx].page_content[:70]}...")

    # --- RRF Fusion ---
    fused = reciprocal_rank_fusion(bm25_top, faiss_top)

    print("\n After RRF Fusion — Final Top 3:")
    final_chunks = []
    for rank, (idx, rrf_score) in enumerate(fused[:top_k]):
        print(f"  Rank {rank+1} | RRF Score: {rrf_score:.6f} | {chunks[idx].page_content[:70]}...")
        final_chunks.append(chunks[idx])

    return final_chunks


# STEP 8: GENERATE ANSWER

def generate_answer(question, relevant_chunks, llm):
    context = "\n\n".join([
        f"[Page {c.metadata.get('page', '?')}]: {c.page_content}"
        for c in relevant_chunks
    ])

    prompt = f"""You are a helpful assistant analyzing a resume/document.
Answer directly and concisely using ONLY the context below.
For resumes: extract names, skills, education, experience directly.
If not found say "Not found in document."

Context:
{context}

Question: {question}

Answer:"""

    return llm.invoke(prompt)


# MAIN


def main():
    pdf_path = "Daivashala_Dhepale_Resume.pdf"      

    # Build pipeline
    pages      = load_pdf(pdf_path)
    chunks     = chunk_documents(pages)
    bm25       = build_bm25(chunks)
    vectorstore = build_faiss(chunks)
    llm        = OllamaLLM(model="llama3.2:1b")

    print("\n Hybrid RAG Chat! Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break

        # Hybrid retrieve
        relevant_chunks = hybrid_retrieve(
            query=question,
            bm25=bm25,
            vectorstore=vectorstore,
            chunks=chunks,
            top_k=3,
            candidate_k=10
        )

        # Generate
        answer = generate_answer(question, relevant_chunks, llm)
        print(f"\n Answer: {answer}\n")


if __name__ == "__main__":
    main()