# ---- Part Three: Turn Summarizer into Q&A Bot--- 
import os 
from dotenv import load_dotenv
from openai import OpenAI
import requests 
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader


# --- Step 1: Initialize ---
# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Embeddings + FAISS
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Path to store FAISS index (memory)
faiss_index_path = "faiss_index"
if os.path.exists(faiss_index_path):
    faiss_store = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
else:
    faiss_store = None


# --- Step 2: Fetch & Clean Article ---
def fetch_article_text(url):
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
    
        for tag in soup(["script","style","noscript", "header", "footer","nav","aside"]):
            tag.decompose()
    
        main_content = soup.find("article")
        if main_content:
            text = main_content.get_text(separator=" ")
        else:
            text = soup.get_text(separator=" ")
        clean_text = " ".join(text.split())
        return clean_text
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return ""
    
# Step 3: Load Local Files
def load_local_file(file_name):
    file_path = os.path.join("data", file_name)

    if file_path.endswith(".txt") or file_path.endswith(".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else: 
        raise ValueError("Unsupported file type. Use .txt, .md, or .pdf")

    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])



# --- Step 4: Split Text into Chunks (for RAG) ---
def split_text(text, chunk_size=500, chunk_overlap=50):
    """Split long text into chunks for embedding and retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


# --- Step 5: Store Chunks in FAISS ---
def store_chunks_in_faiss(chunks, embedding_model, faiss_index_path, metadata=None):
    """Store or update chunks in FAISS vector DB."""
    if os.path.exists(faiss_index_path):
        # Load existing index
        faiss_store = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
        # add new chunks into it
        faiss_store.add_texts(chunks, metadatas=metadata or [{}])
    else:
        # Create new index from scratch
        faiss_store = FAISS.from_texts(chunks, embedding_model,metadatas=metadata or [{}])
    
    faiss_store.save_local(faiss_index_path)
    return faiss_store

# --- Step 6: Q&A Function ---
def answer_question_with_memory(question, faiss_store, client, chat_history, k=3):
    # Retrieve relevant document chunks
    query_vector = faiss_store.embeddings.embed_query(question)
    docs = faiss_store.similarity_search_by_vector(query_vector, k=k)
    retrieved_chunks = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

    # Include previous Q&A in context (last 3 exchanges, for example)
    conversation_context = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in chat_history[-3::]])
    

    # Propmt GPT
    propmt_template = PromptTemplate(
        input_variables=["retrieved_chunks","conversation_context","query"],
        template="""You are a helpful assistant. Use the provided context from documents and previous conversation to answer the question. Context from documents: {retrieved_chunks}

        Previous conversation:
        {conversation_context}

        New question:
        {query}"""
    )

    prompt = propmt_template.format(
        retrieved_chunks=retrieved_chunks, 
        conversation_context=conversation_context or "None",
        query=question
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )

    answer = response.choices[0].message.content.strip()

    # Save Q&A to memory
    chat_history.append({"question": question, "answer": answer})

    return answer

# --- Step 7: CLI ---
if __name__ == "__main__":
    choice = input("Load from (1) URL or (2) File? ").strip()
    if choice == "1":
        url = input("Enter article URL: ").strip()
        text = fetch_article_text(url)
        metadata= {"source_url":url}

    elif choice == "2":
        file_path = input("Enter file path (.txt, .md, .pdf): ").strip()
        text = load_local_file(file_path)
        metadata = {"source_url":file_path}

    else:
        print("Invalid choice. Exiting")
        exit()

    if not text.strip():
        print("Error: No Text provided")
    
    # Split and store chunks in FAISS for RAG
    chunks = split_text(text)

    # Expand metadata so each chunk has one
    metadata_list = [metadata for _ in range(len(chunks))]
    # Create metadata per chunk
    faiss_store = store_chunks_in_faiss(chunks, embedding_model, faiss_index_path, metadata_list)

    print("\nRAG setup complete. Ask your questions!")

    # Q&A loop
    # Collect chat history
    chat_history = []
    while True:
        question = input("\nAsk a question (or 'exit'): ").strip()
        if question.lower() in ["exit", "quit"]:
            break
        answer = answer_question_with_memory(question, faiss_store, client, chat_history)
        print("\nAnswer:", answer)



