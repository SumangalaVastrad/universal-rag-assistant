import os
import re
import time
import math
from collections import Counter
import numpy as np

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class FastFallbackEmbedder:
    """Fast TF-IDF / character-word n-gram fallback embedder when SentenceTransformers is not yet loaded."""
    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            words = re.findall(r"\w+", text.lower())
            vec = np.zeros(self.dim, dtype=np.float32)
            if not words:
                embeddings.append(vec)
                continue
            for i, w in enumerate(words):
                h = hash(w) % self.dim
                vec[h] += 1.0 / (1.0 + math.log(1 + i))
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)


class RagEngine:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.faiss_index = None
        self.embedding_dim = 384
        self.embed_model = None
        self.audit_log = []

        # Load SentenceTransformer model
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                print("[RagEngine] Loading embedding model 'all-MiniLM-L6-v2'...")
                self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedding_dim = 384
            except Exception as e:
                print(f"[RagEngine] Could not load SentenceTransformer: {e}. Using fallback embedder.")
                self.embed_model = FastFallbackEmbedder(self.embedding_dim)
        else:
            print("[RagEngine] sentence-transformers not installed. Using fallback embedder.")
            self.embed_model = FastFallbackEmbedder(self.embedding_dim)

        self._init_vector_store()

        # Groq client
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.groq_client = None
        if self.groq_api_key and HAS_GROQ:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                print(f"[RagEngine] Groq client init failed: {e}")

    def _init_vector_store(self):
        if HAS_FAISS:
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            self.faiss_index = None

    def _extract_text_and_pages(self, filepath):
        """Extract text with page numbers from PDF, DOCX, or TXT."""
        pages = []
        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".pdf" and HAS_PYPDF:
            reader = PdfReader(filepath)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((i + 1, text))
        elif ext == ".docx" and HAS_DOCX:
            doc = docx.Document(filepath)
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            pages.append((1, full_text))
        elif ext == ".csv":
            try:
                import pandas as pd
                df = pd.read_csv(filepath)
                cols = list(df.columns)
                rows_text = []
                for idx, row in df.iterrows():
                    row_desc = ", ".join([f"{col}: {row[col]}" for col in cols if pd.notna(row[col])])
                    rows_text.append(f"Row {idx+1} -> {row_desc}")
                
                chunk_size = 15
                if not rows_text:
                    pages.append((1, f"CSV File with Columns: {', '.join(map(str, cols))} (No data rows)"))
                else:
                    for page_idx, i in enumerate(range(0, len(rows_text), chunk_size), 1):
                        batch = "\n".join(rows_text[i : i + chunk_size])
                        pages.append((page_idx, f"CSV File Columns: {', '.join(map(str, cols))}\n\nData:\n{batch}"))
            except Exception as e:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    pages.append((1, f.read()))

        elif ext in [".xlsx", ".xls"]:
            try:
                import pandas as pd
                excel_file = pd.ExcelFile(filepath)
                for sheet_idx, sheet_name in enumerate(excel_file.sheet_names, 1):
                    df = pd.read_excel(filepath, sheet_name=sheet_name)
                    cols = list(df.columns)
                    rows_text = []
                    for idx, row in df.iterrows():
                        row_desc = ", ".join([f"{col}: {row[col]}" for col in cols if pd.notna(row[col])])
                        rows_text.append(f"Row {idx+1} -> {row_desc}")
                    
                    chunk_size = 15
                    if not rows_text:
                        pages.append((sheet_idx, f"Excel Sheet: '{sheet_name}' | Columns: {', '.join(map(str, cols))} (Empty Sheet)"))
                    else:
                        for sub_idx, i in enumerate(range(0, len(rows_text), chunk_size), 1):
                            batch = "\n".join(rows_text[i : i + chunk_size])
                            pages.append((sheet_idx, f"Excel Sheet: '{sheet_name}' | Columns: {', '.join(map(str, cols))}\n\nData:\n{batch}"))
            except Exception as e:
                pages.append((1, f"Excel file: {os.path.basename(filepath)}"))

        else:
            encodings = ["utf-8", "latin-1", "cp1252"]
            content = ""
            for enc in encodings:
                try:
                    with open(filepath, "r", encoding=enc) as f:
                        content = f.read()
                    break
                except Exception:
                    continue
            pages.append((1, content))

        return pages

    def _chunk_text(self, text, max_words=120, overlap_words=25):
        """Split text into overlapping chunks."""
        words = text.split()
        if not words:
            return []
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + max_words])
            chunks.append(chunk)
            i += max_words - overlap_words
        return chunks

    def ingest_document(self, filepath, doc_role="public"):
        """Ingest document, chunk text, generate embeddings, and update index."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        filename = os.path.basename(filepath)
        pages = self._extract_text_and_pages(filepath)
        if not pages:
            raise ValueError(f"Could not extract any text from {filename}")

        new_chunks = []
        new_texts = []
        is_stale_file = "stale" in filename.lower() or "archive" in filename.lower()
        chunk_timestamp = (time.time() - 200 * 86400) if is_stale_file else time.time()

        for page_num, page_text in pages:
            text_chunks = self._chunk_text(page_text)
            for t_chunk in text_chunks:
                chunk_obj = {
                    "text": t_chunk,
                    "source": filename,
                    "page": page_num,
                    "doc_role": doc_role,
                    "timestamp": chunk_timestamp,
                    "images": [],
                }
                new_chunks.append(chunk_obj)
                new_texts.append(t_chunk)

        if not new_texts:
            raise ValueError(f"No valid text chunks generated from {filename}")

        # Compute embeddings
        embs = self.embed_model.encode(new_texts)
        if isinstance(embs, list):
            embs = np.array(embs, dtype=np.float32)
        elif not isinstance(embs, np.ndarray):
            embs = embs.cpu().numpy().astype(np.float32)
        else:
            embs = embs.astype(np.float32)

        # Update Vector Index
        if HAS_FAISS and self.faiss_index is not None:
            self.faiss_index.add(embs)

        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = embs
        else:
            self.embeddings = np.vstack([self.embeddings, embs])

        self.chunks.extend(new_chunks)

    def get_stats(self):
        sources = sorted(list(set(c["source"] for c in self.chunks)))
        return {
            "total_chunks": len(self.chunks),
            "unique_sources": len(sources),
            "sources": sources,
        }

    def _keyword_search(self, query_tokens, candidate_indices):
        """Source-aware keyword search returning ranked candidate indices."""
        scores = {}
        for idx in candidate_indices:
            chunk = self.chunks[idx]
            text = chunk["text"].lower()
            source = chunk["source"].lower()
            score = 0
            for token in query_tokens:
                if token in source:
                    score += 3.0
                if token in text:
                    score += 1.0
            if score > 0:
                scores[idx] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in ranked]

    def query(self, message, top_k=8, user_role=None):
        """Hybrid search combining semantic search & keyword search with Reciprocal Rank Fusion (RRF)."""
        if not self.chunks:
            return []

        candidate_indices = list(range(len(self.chunks)))

        # 1. Semantic Search
        q_emb = self.embed_model.encode([message])
        if isinstance(q_emb, list):
            q_emb = np.array(q_emb, dtype=np.float32)
        elif not isinstance(q_emb, np.ndarray):
            q_emb = q_emb.cpu().numpy().astype(np.float32)
        else:
            q_emb = q_emb.astype(np.float32)

        cand_embs = self.embeddings[candidate_indices]
        dists = np.linalg.norm(cand_embs - q_emb, axis=1)
        sem_ranked_order = np.argsort(dists)
        sem_ranked_indices = [candidate_indices[i] for i in sem_ranked_order]

        # 2. Keyword Search
        query_tokens = [
            t.lower()
            for t in re.findall(r"\w+", message)
            if len(t) > 2 and t.lower() not in {"what", "how", "why", "when", "where", "the", "and", "for"}
        ]
        kw_ranked_indices = self._keyword_search(query_tokens, candidate_indices)

        # 3. Reciprocal Rank Fusion (RRF, k=60)
        rrf_scores = Counter()
        for rank, idx in enumerate(sem_ranked_indices):
            rrf_scores[idx] += 1.0 / (60.0 + rank + 1)

        for rank, idx in enumerate(kw_ranked_indices):
            rrf_scores[idx] += 1.0 / (60.0 + rank + 1)

        top_candidates = rrf_scores.most_common(top_k)

        results = []
        retrieved_sources = set()
        for idx, score in top_candidates:
            chunk = self.chunks[idx]
            retrieved_sources.add(chunk["source"])
            results.append({
                "source": chunk["source"],
                "page": chunk.get("page", 1),
                "score": float(score),
                "doc_role": chunk.get("doc_role", "public"),
                "images": chunk.get("images", []),
                "text": chunk["text"],
            })

        # Record in audit log
        self.audit_log.append({
            "timestamp": time.time(),
            "question": message,
            "sources_retrieved": sorted(list(retrieved_sources)),
        })

        return results

    def generate_answer(self, message, chunks):
        """Generate direct, beautifully formatted answer with citations using Groq Qwen 3.6 27B."""
        if not chunks:
            return "No matching context found."

        # Prepare context blocks with source numbering
        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {c['source']} | Page {c['page']}]\n{c['text']}"
            )
        context_str = "\n\n".join(context_parts)

        api_key = os.environ.get("GROQ_API_KEY", self.groq_api_key or "").strip()
        model_name = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b").strip()

        if api_key:
            candidate_models = [model_name, "qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "groq/compound"]
            # remove duplicates while preserving order
            unique_models = []
            for m in candidate_models:
                if m not in unique_models:
                    unique_models.append(m)

            for target_model in unique_models:
                try:
                    system_prompt = (
                        "You are the Universal RAG AI Knowledge Assistant. "
                        "Your goal is to provide rich, comprehensive, beautifully structured answers strictly grounded in the provided context.\n\n"
                        "FORMATTING RULES:\n"
                        "1. When answering about resumes/candidates: Clearly separate each candidate using headers like '👤 Candidate: [Name]'. "
                        "Organize their details into '📚 Technical Skills & Core Tools [Source X, Page Y]' (grouped by Languages, Databases, Cloud, Frameworks) "
                        "and '📁 Main Projects & Associated Technologies' or '🔹 Project X: [Title]' with 'Technologies Used' and 'Key Contributions & Details'.\n"
                        "2. When answering about tabular CSV/Excel files: Extract specific rows, values, metrics, and comparisons in clean Markdown tables and bullet points.\n"
                        "3. When answering about research papers or technical reports: Clearly break down methodology, pipeline, dataset, and key benchmarks/metrics.\n"
                        "4. When referencing information, always cite the source (e.g., '[Source 1, Page 1]' or 'Source: Source 1, Page 1').\n"
                        "5. Maintain high fidelity to the text (exact libraries, versions, metrics, algorithms).\n"
                        "6. End with a polite, helpful follow-up note (e.g., 'Let me know if you need a comparison with other candidates or deeper technical breakdowns!')."
                    )
                    user_prompt = (
                        f"CONTEXT INFORMATION:\n{context_str}\n\n"
                        f"USER QUESTION: {message}\n\n"
                        f"Provide a structured, complete answer following the formatting rules."
                    )

                    if HAS_GROQ:
                        client = self.groq_client or Groq(api_key=api_key)
                        try:
                            completion = client.chat.completions.create(
                                model=target_model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt},
                                ],
                                temperature=0.2,
                                max_tokens=850,
                                reasoning_format="hidden",
                            )
                        except Exception:
                            completion = client.chat.completions.create(
                                model=target_model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt},
                                ],
                                temperature=0.2,
                                max_tokens=850,
                            )
                        raw_content = completion.choices[0].message.content or ""
                        clean_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
                        if clean_content or raw_content:
                            return clean_content or raw_content

                except Exception as e:
                    print(f"[RagEngine] Model {target_model} error: {e}")
                    continue

        # Fallback local synthesis
        synthesis = "Based on the relevant documents indexed in the knowledge base:\n\n"
        for i, c in enumerate(chunks[:3], 1):
            synthesis += f"> **[{c['source']} - Page {c['page']}]**\n> \"{c['text'][:280]}...\"\n\n"
        return synthesis

    def get_audit_summary(self):
        """Governance summary: chunks by role/department, staleness, and query audit log."""
        chunks_by_role = Counter([c.get("doc_role", "public") for c in self.chunks])
        now = time.time()
        stale_count = sum(1 for c in self.chunks if (now - c.get("timestamp", now)) > (180 * 86400))
        return {
            "chunks_by_role": dict(chunks_by_role),
            "total_queries": len(self.audit_log),
            "stale_chunk_count": stale_count,
            "recent_log": self.audit_log[-15:],
        }

    def clear_index(self):
        """Clear all stored chunks and reset vector index."""
        self.chunks = []
        self.embeddings = None
        self._init_vector_store()
        print("[RagEngine] Vector index and chunks cleared.")
