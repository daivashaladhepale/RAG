
# BM25 RAG - Chat with PDF using Keyword Search (No Vectors!)


import re
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaLLM



# STEP 1: LOAD PDF

def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():                   
            pages.append({
                "text": text,
                "page": i
            })
    print(f" Loaded {len(pages)} pages")
    return pages



# STEP 2: CHUNK THE TEXT
# (same as dense RAG — split into small pieces)

def chunk_documents(pages, chunk_size=500, overlap=50):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({
                "text": chunk_text,
                "page": page["page"]
            })
            start += chunk_size - overlap   
    print(f" Created {len(chunks)} chunks")
    return chunks



# STEP 3: TOKENIZE
# BM25 doesn't use vectors — it just splits text into words

def tokenize(text):
    
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)  # extract only words
    return tokens




# STEP 4: BUILD BM25 INDEX
# (like FAISS but for keywords, not vectors)

def build_bm25_index(chunks):
    # Tokenize every chunk
    tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]

    # BM25Okapi builds the index from tokenized chunks
    bm25 = BM25Okapi(tokenized_chunks)

    print(" BM25 index built")

    
    print("\n Sample tokenized chunk:")
    print(f"   Original : {chunks[0]['text'][:80]}...")
    print(f"   Tokenized: {tokenized_chunks[0][:15]}...")  # first 15 tokens

    return bm25, tokenized_chunks



# STEP 5: RETRIEVE
# (score all chunks against query, return top K)

def retrieve(query, bm25, chunks, top_k=3):
    # Tokenize the query the same way
    query_tokens = tokenize(query)
    print(f"\n Query tokens: {query_tokens}")

    # BM25 scores every chunk
    scores = bm25.get_scores(query_tokens)

    # Sort by score descending
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    #  Show scores so you can see how BM25 works
    print("\n BM25 Scores (top 3):")
    for rank, idx in enumerate(top_indices):
        print(f"   Rank {rank+1} | Score: {scores[idx]:.4f} | "
              f"Page {chunks[idx]['page']} | "
              f"{chunks[idx]['text'][:60]}...")

    return [chunks[i] for i in top_indices]



# STEP 6: Generate ans with llm

def generate_answer(question, relevant_chunks, llm):
    # Join retrieved chunks as context
    context = "\n\n".join([
        f"[Page {c['page']}]: {c['text']}"
        for c in relevant_chunks
    ])

    prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer.
If the answer isn't in the context, say "I don't know based on the document."
Give a direct, concise answer. Do NOT repeat the context.

Context:
{context}

Question: {question}

Answer:"""

    return llm.invoke(prompt)



# MAIN

def main():
    pdf_path = "Daivashala_Dhepale_Resume.pdf"       

    # Build pipeline
    pages    = load_pdf(pdf_path)
    chunks   = chunk_documents(pages)
    bm25, _  = build_bm25_index(chunks)
    llm      = OllamaLLM(model="llama3.2:1b")

    print("\n BM25 Chat with PDF! Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break

        # Retrieve relevant chunks
        relevant_chunks = retrieve(question, bm25, chunks, top_k=3)

        # Generate answer
        answer = generate_answer(question, relevant_chunks, llm)
        print(f"\n Answer: {answer}\n")


if __name__ == "__main__":
    main()