# ---- Part 4 — Multi-File Support + Chunking--- 
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

# --- Step 4: Summarization ---
def summarize_text(text, client, style="concise"):
    """Summarize the text using GPT-4o-mini."""
    prompt_template = PromptTemplate(
        input_variables=["style","content"],
        template="Summarize the following text in a {style} way: \n\n{content}"
    )
    prompt = prompt_template.format(style=style, content=text[:12000])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user","content": prompt}],
        temperature=0.5,
        max_tokens=250
    )
    return response.choices[0].message.content.strip()


# --- Step 5: Split Text into Chunks (for RAG) ---
def split_text(text, chunk_size=500, chunk_overlap=50):
    """Split long text into chunks for embedding and retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


# --- Step 6: Store Chunks in FAISS ---
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

# --- Step 7: Q&A Function ---
def answer_question(question, faiss_store, client, k=3):
    """Answer a user question using RAG with FAISS memory and GPT."""
    embedding_model = faiss_store.embeddings
    query_vector = embedding_model.embed_query(question)

    # retrieve relevenat chunks from FAISS
    docs = faiss_store.similarity_search_by_vector(query_vector, k=k)
    
    if not docs:
        return "Sorry, I couldn't find anything in the article related to your question."

  
    context = "\n\n".join([doc.page_content for doc in docs])

    # Ask GPT using the retrieved context
    prompt = f"""
    You are a highly intelligent assistant specializing in answering questions based strictly on provided text excerpts. 
    Use ONLY the information in the context to answer the question. 
    Be concise, factual, and clear. 
    Do not include any information not found in the context.
    If the answer is not in the text, respond with 'I don't know'. 
    Provide bullet points if it helps clarify the answer.

    Context:
    {context}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content": prompt},
            {"role":"user", "content": question},
          
            ],
        temperature=0.2,
        max_tokens=250
    )
    return response.choices[0].message.content.strip()

# --- Step 7: CLI ---
if __name__ == "__main__":
    choice = input("Do you want to load from (1) URL or (2) File? ").strip()
    style = input("Summary style (concise/detailed/bullets/simple): ").strip().lower() or "concise"
    if choice == "1":
        url = input("Enter article URL: ").strip()
        text = fetch_article_text(url)
        source_info = {"source_url":url}

    elif choice == "2":
        file_path = input("Enter file path (.txt, .md, .pdf): ").strip()
        text = load_local_file(file_path)
        source_info =  {"source_url":file_path}

    else:
        print("Invalid choice. Exiting")
        exit()

    if not text.strip():
        print("Error: No Text provided")
    
        # Summarize first
    summary = summarize_text(text, client, style)
    print("\n=== Summary ===\n")
    print(summary)

    # Split and store chunks in FAISS for RAG
    chunks = split_text(text)
    # Create metadata per chunk
    metadata = [source_info for _ in range(len(chunks))]
    faiss_store = store_chunks_in_faiss(chunks, embedding_model, faiss_index_path, metadata)

         # Q&A loop
    while True:
        question = input("\nAsk a question about the article (or 'exit'): ").strip()
        if question.lower() in ["exit", "quit"]:
            break
        answer = answer_question(question, faiss_store, client)
        print("\nAnswer:", answer)



