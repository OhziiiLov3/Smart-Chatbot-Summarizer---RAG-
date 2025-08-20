# Smart Chatbot(Summarizer -> RAG)
## End-to-End RAG Pipeline

###  Goal
Build a pipeline where your bot can:

1. Load multiple file types (`.txt`, `.md`, `.pdf`) from a folder or fetch articles from a URL.  
2. Chunk documents into manageable pieces.  
3. Store chunks in **FAISS** with metadata.  
4. Answer user questions using **retrieval-augmented generation (RAG)**.

### Steps
1. **Load Documents**  
   - From `/data` folder or a URL.
2. **Summarize & Chunk**  
   - Split text into 500–1000 character chunks.  
   - Attach metadata (filename, type, chunk ID).
3. **Store in FAISS**  
   - Create or update FAISS vector store.
4. **User Query / RAG**  
   - Embed the query using OpenAI Embeddings.  
   - Retrieve top-k relevant chunks from FAISS.  
   - Inject chunks into LLM prompt.
5. **Generate Answer**  
   - GPT generates answer based strictly on retrieved context.
   
### ✅ Outcome
Users can ask questions and receive **context-aware answers** sourced from multiple documents, rather than relying on raw memory alone.


```mermaid
flowchart TD

    %% --- Document Ingestion ---
    A[Start] --> B{Input Source?}
    B -->|URL| C[Fetch Article Text with BeautifulSoup]
    B -->|Files| D[Load Files from /data folder - txt, md, pdf]

    C --> E[Summarize Text with GPT]
    D --> E

    %% --- Chunking & FAISS ---
    E --> F[Split into Chunks 500-1000 chars]
    F --> G[Attach Metadata - filename, type, chunk ID]

    G --> H{FAISS Index Exists?}
    H -->|Yes| I[Load FAISS Index and Add Chunks]
    H -->|No| J[Create New FAISS Index from Chunks]

    I --> K[Save Updated Index]
    J --> K

    %% --- RAG Pipeline ---
    K --> L[User Query]
    L --> M[Embed Query using OpenAI Embeddings]
    M --> N[Retrieve Relevant Chunks from FAISS]
    N --> O[Inject Chunks into LLM Prompt]
    O --> P[GPT Generates Answer]
    P --> Q[Return Answer to User]

    Q --> L[Loop for next query]

```