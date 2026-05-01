# Universal Multi-Document Hybrid RAG Engine

A retrieval-augmented generation (RAG) system built with **Python**, **LangChain**, **FAISS**, and **Groq LPU**, designed to ingest and query heterogeneous document collections (research papers, technical reports, tabular data, and resumes) with grounded source citations.

---

## Architecture Overview

```
                                      +--------------------+
                                      | User Query & Role  |
                                      +---------+----------+
                                                |
                 +------------------------------+------------------------------+
                 |                                                             |
                 v                                                             v
       +-------------------+                                         +-------------------+
       | Dense Retriever   |                                         | Lexical Retriever |
       | (FAISS + MiniLM)  |                                         | (BM25 Keyword)    |
       +---------+---------+                                         +---------+---------+
                 |                                                             |
                 +------------------------------+------------------------------+
                                                |
                                                v
                               +---------------------------------+
                               |  Reciprocal Rank Fusion (RRF)   |
                               |          k = 60 Score           |
                               +----------------+----------------+
                                                |
                                                v Top-k Context
                               +---------------------------------+
                               |   Groq LPU Inference Engine     |
                               | (Qwen 2.5 / Llama 3 70B Stream) |
                               +----------------+----------------+
                                                |
                                                v Grounded Answer
                               +---------------------------------+
                               | Gradio UI + Page Citations & QA |
                               +---------------------------------+
```

### Key Technical Highlights
- **Hybrid Retrieval (Dense + Lexical)**: Fuses embedding similarity from `sentence-transformers/all-MiniLM-L6-v2` with lexical BM25 scores using Reciprocal Rank Fusion (RRF, \(k=60\)), preventing semantic drift on keyword-exact entity queries.
- **Page-Aware Ingestion Pipeline**: Preserves bounding page metadata across PDF, DOCX, TXT, CSV, and Excel tables so every generated claim links back to its verified source and page index.
- **Low-Latency Inference**: Integrates Groq LPU API for token generation rates (>250 tokens/sec), with built-in retry and fallback logic.
- **Departmental Access Governance**: Role-based access control (Admin, HR, Finance, Engineering) controlling which document indexes can be searched or modified.

---

## Repository Structure

```
universal-rag-assistant/
├── app.py                      # Gradio web user interface and session manager
├── rag_engine.py               # Document loaders, chunking, FAISS index, and RRF logic
├── requirements.txt            # Project dependencies
├── render.yaml                 # Deployment specification for cloud hosting
├── sample_docs/                # Test documents for validation
├── tests/
│   ├── test_rag.py             # Vector store and retrieval precision tests
│   ├── test_csv_excel.py       # Tabular data ingestion and schema tests
│   └── test_upload_permissions.py # Role-based access control tests
└── README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or 3.11
- A free Groq API key from [Groq Console](https://console.groq.com/)

### Step 1: Clone the repository
```bash
git clone https://github.com/SumangalaVastrad/universal-rag-assistant.git
cd universal-rag-assistant
```

### Step 2: Set up virtual environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure environment variables
Create a `.env` file in the root directory:
```ini
GROQ_API_KEY=your_groq_api_key_here
EMBEDDING_MODEL=all-MiniLM-L6-v2
RRF_K=60
CHUNK_SIZE=512
CHUNK_OVERLAP=64
```

---

## Running the Application

Launch the local interactive Gradio interface:
```bash
python app.py
```
Open `http://localhost:7860` in your browser.

1. **Upload Documents**: Drag and drop PDF, DOCX, or CSV files in the sidebar.
2. **Indexing**: The engine extracts text, computes embeddings, and builds the FAISS index in memory.
3. **Ask Questions**: Query the indexed corpus. Answers include collapsible citations showing the exact snippet and page number.

---

## Testing & Validation

Run the automated test suite with `pytest`:
```bash
pytest test_rag.py -v
pytest test_csv_excel.py -v
pytest test_upload_permissions.py -v
```

---

## Author
**Sumangala Vastrad**  
B.E. in Artificial Intelligence & Data Science, Nitte Meenakshi Institute of Technology (NMIT)  
Email: [sbvastrad2005@gmail.com](mailto:sbvastrad2005@gmail.com)  
GitHub: [@SumangalaVastrad](https://github.com/SumangalaVastrad)
