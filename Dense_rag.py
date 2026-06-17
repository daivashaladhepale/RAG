
# CHAT WITH PDF - Full RAG Pipeline using Ollama (Local/Free)


# IMPORTS 
from langchain_community.document_loaders import PyPDFLoader       # reads PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter # chunking
from langchain_ollama import OllamaEmbeddings                       # embeddings via ollama
from langchain_community.vectorstores import FAISS                  # vector DB
from langchain_ollama import OllamaLLM          
from langchain_core.output_parsers import StrOutputParser                    # LLM via ollama
from langchain_core.prompts import PromptTemplate    
from langchain_core.runnables import RunnablePassthrough                    # custom prompt



# STEP 1: LOAD THE PDF

def load_pdf(path):
    loader = PyPDFLoader(path)      
    pages = loader.load()           
    print(f" Loaded {len(pages)} pages")
    return pages



# STEP 2: CHUNK THE TEXT

def chunk_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # each chunk = max 500 characters
        chunk_overlap=50,     # 50 char overlap between chunks (preserves context)
        length_function=len,
    )
    chunks = splitter.split_documents(pages)
    print(f" Created {len(chunks)} chunks")
    return chunks



# STEP 3 + 4: EMBED AND STORE IN FAISS

def create_vectorstore(chunks):
    # OllamaEmbeddings uses your local ollama to generate vectors
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # FAISS takes all chunks, embeds them, stores them
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(" Vector store created")
    return vectorstore



# STEP 5 + 6: RETRIEVE + GENERATE (The QA Chain)

def create_qa_chain(vectorstore):
    llm = OllamaLLM(model="llama3.2:1b")

    prompt = PromptTemplate.from_template("""
    You are a helpful assistant answering questions about a document.

    Rules:
    - If the question is a greeting like "hi", "hello" etc, just greet back normally.
    - Use ONLY the context below to answer document questions.
    - Give direct, concise answers. Do NOT repeat the context back.
    - If the answer isn't in the context, say "I don't know based on the document."

    Context:
    {context}

    Question: {question}

    Answer (be concise and direct):""")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # way to build chains in LangChain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def main():
    pdf_path = "Daivashala_Dhepale_Resume.pdf"   

    pages = load_pdf(pdf_path)
    chunks = chunk_documents(pages)
    vectorstore = create_vectorstore(chunks)
    chain, retriever = create_qa_chain(vectorstore)

    print("\n Chat with your PDF! Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break

        # Get answer
        answer = chain.invoke(question)
        print(f"\n Answer: {answer}")

        # Show source chunks
        docs = retriever.invoke(question)
        print("\n Sources used:")
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] Page {doc.metadata.get('page', '?')}: {doc.page_content[:100]}...")
        print()

if __name__ == "__main__":
    main()