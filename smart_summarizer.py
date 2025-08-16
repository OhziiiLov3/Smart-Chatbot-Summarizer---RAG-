import os 
from dotenv import load_dotenv
from openai import OpenAI
import requests 
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# Step 1: Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Step 2: Fetch & clean article text
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

# Step 3: Initialize Embeddings + FAISS
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Load or create FAISS index
faiss_index_path = "faiss_index"
if os.path.exists(faiss_index_path):
    faiss_store = FAISS.load_local(faiss_index_path, embedding_model, allow_danerous_deserializetion=True)
else:
    faiss_store = None


# Step 4: Summarize with memory
def summarize_text(text,client, embedding_model, faiss_store, faiss_index_path, style="concise"):
# 4a. embed input text 
    query_vector = embedding_model.embed_query(text[:1000])

# 4b. Check FAISS for similar summaries
    if faiss_store:
        docs = faiss_store.similarity_search_by_vector(query_vector,k=1)
        if docs:
            print("Found similar sumary in memory. Returning cached result.")
            return docs[0].page_content, faiss_store
    
# 4c. otherwise, call GPT
    prompt_template = PromptTemplate(
        input_variables = ["style", "content"],
        template="Summarize the following text in a {style} way: \n\n{content}"
    )
    prompt = prompt_template.format(style=style, content=text[:12000])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user", "content":prompt}],
        temperature=0.5,
        max_tokens=250
    )
    summary = response.choices[0].message.content.strip()
# 4d. save the new summary in FAISS
    if faiss_store is None:
        faiss_store = FAISS.from_texts([summary], embeddings=embedding_model, metadatas=[{"similarity":1.0}])
    else:
        faiss_store.ADD_TEXTS([summary], metadatas=[{"source": "summary"}])

    faiss_store.save_local(faiss_index_path)
    return summary , faiss_store


# ------- CLI -------
if __name__ == "__main__":
    url = input("Enter article URL: ").strip()
    style = input("Summary style (concise/detailed/bullets): ").strip().lower() or "concise"

    text = fetch_article_text(url)
    if not text.strip():
        print("Error: No text provided.")
    else:
        summary, faiss_store = summarize_text(
            text,
            client,
            embedding_model,
            faiss_store,
            faiss_index_path,
            style
        )
        print("\n=== Summary ===\n")
        print(summary)