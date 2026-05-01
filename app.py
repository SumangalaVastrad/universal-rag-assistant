import os
import time

# Force CPU mode and prevent HF Spaces ZeroGPU CUDA conflict
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Disable Gradio 6 Server-Side Rendering (causes blank page on HF Spaces)
os.environ["GRADIO_SSR_MODE"] = "false"

# Handle spaces import for HF Spaces compatibility
try:
    import spaces
except ImportError:
    class _MockSpaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is not None:
                return func
            def decorator(f):
                return f
            return decorator
    spaces = _MockSpaces()

@spaces.GPU
def _gpu_init():
    """Dummy GPU function required by HF Spaces ZeroGPU infrastructure."""
    return "GPU Initialized"

try:
    _gpu_init()
except Exception as e:
    print(f"[ZeroGPU] Init call note: {e}")

import gradio as gr

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rag_engine import RagEngine

try:
    rag_engine = RagEngine()
except Exception as e:
    print(f"[ERROR] Failed to initialize RAG engine: {e}")
    rag_engine = None

UPLOAD_ROLES = [
    "Admin (Full Access)",
    "HR Department Head",
    "Finance Department Head",
    "Staff / General User",
]


# ════════════════════════════════════════════════════════
# UPLOAD PERMISSION & INGESTION GOVERNANCE
# ════════════════════════════════════════════════════════

DOCUMENT_TYPES = [
    "HR Document",
    "Finance Document",
    "Admin Document",
    "Public / General Document",
]

# Map dropdown label → internal key
_DOC_TYPE_MAP = {
    "HR Document": "hr",
    "Finance Document": "finance",
    "Admin Document": "admin",
    "Public / General Document": "public",
}

# Which doc types each role is allowed to upload
_ROLE_ALLOWED_TYPES = {
    "admin":   {"hr", "finance", "admin", "public"},
    "hr":      {"hr", "public"},
    "finance": {"finance", "public"},
    "staff":   {"public"},
}

_ROLE_LABELS = {
    "admin":   "Admin (Full Access)",
    "hr":      "HR Department Head",
    "finance": "Finance Department Head",
    "staff":   "Staff / General User",
}


def _get_role_key(uploader_role: str) -> str:
    if "HR" in uploader_role:
        return "hr"
    if "Finance" in uploader_role:
        return "finance"
    if "Staff" in uploader_role or "General" in uploader_role:
        return "staff"
    return "admin"


def _validate_upload_permission(role_key: str, doc_type: str) -> tuple[bool, str]:
    """
    Validate whether the uploader has the necessary security authorization
    to ingest the given document type.

    Rules:
    - Admin       : Full access (HR, Finance, Admin, Public).
    - HR Head     : HR + Public only.  Blocked from Finance / Admin.
    - Finance Head: Finance + Public only.  Blocked from HR / Admin.
    - Staff       : Public only.  Blocked from all sensitive docs.
    """
    allowed = _ROLE_ALLOWED_TYPES.get(role_key, set())
    if doc_type in allowed:
        return True, f"Authorized ({_ROLE_LABELS[role_key]})"
    return (
        False,
        f"⛔ Security Policy Violation: "
        f"**{_ROLE_LABELS[role_key]}** is NOT authorized to upload "
        f"**{doc_type.upper()}** documents. "
        f"This role may only upload: {', '.join(t.upper() for t in sorted(allowed))}."
    )


def ingest_file(files, uploader_role, declared_doc_type_label):
    """
    Handle uploaded file(s): enforce strict departmental upload permissions
    using the EXPLICITLY DECLARED document type (not guessed from filename).
    """
    if rag_engine is None:
        return "❌ RAG engine not initialized. Check GROQ_API_KEY secret."
    if not files:
        return "No file uploaded."

    doc_type = _DOC_TYPE_MAP.get(declared_doc_type_label, "public")
    role_key = _get_role_key(uploader_role)

    is_allowed, reason = _validate_upload_permission(role_key, doc_type)

    if not isinstance(files, list):
        files = [files]

    ingested_names = []
    rejected_files = []
    errors = []

    for file in files:
        filepath = file.name if hasattr(file, "name") else file
        filename = os.path.basename(filepath)

        if not is_allowed:
            rejected_files.append(f"{filename} [{doc_type.upper()} Document] → {reason}")
            continue

        try:
            rag_engine.ingest_document(filepath, doc_role=doc_type)
            ingested_names.append(f"{filename} [{doc_type.upper()}] (Verified by {uploader_role})")
        except Exception as e:
            errors.append(f"{filename}: {e}")

    stats = rag_engine.get_stats()

    status_parts = []
    if ingested_names:
        status_parts.append(
            f"✅ **Successfully Ingested ({len(ingested_names)} file(s)):**\n" +
            "\n".join(f" * {name}" for name in ingested_names)
        )
    if rejected_files:
        status_parts.append(
            f"⛔ **Upload Access Control Blocked ({len(rejected_files)} file(s)):**\n" +
            "\n".join(f" * {rej}" for rej in rejected_files)
        )
    if errors:
        status_parts.append(
            f"⚠️ **Failed to Ingest:**\n" +
            "\n".join(f" * {err}" for err in errors)
        )

    status_parts.append(
        f"📊 **Index Status:** Total chunks: **{stats['total_chunks']}** | Unique sources: **{stats['unique_sources']}**"
    )
    return "\n\n".join(status_parts)


# ════════════════════════════════════════════════════════
# AURORA MIDNIGHT CYBER-GLASS THEME (CSS)
# ════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
  --primary-cyan: #06b6d4; --primary-indigo: #6366f1; --primary-purple: #a855f7;
  --aurora-gradient: linear-gradient(135deg, #06b6d4 0%, #6366f1 50%, #a855f7 100%);
  --bg-dark: #050811; --surface-glass: rgba(13, 19, 33, 0.75); --border-glow: rgba(99, 102, 241, 0.25);
}
body, gradio-app {
  background-color: #050811 !important;
  background-image: radial-gradient(at 15% 15%, rgba(6, 182, 212, 0.08) 0px, transparent 50%),
                     radial-gradient(at 85% 85%, rgba(168, 85, 247, 0.08) 0px, transparent 50%) !important;
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  color: #f1f5f9 !important;
}
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; padding: 24px 16px !important; }
.hero-banner {
  text-align: center; padding: 34px 24px 26px 24px; margin-bottom: 24px;
  background: radial-gradient(circle at 50% -20%, rgba(99, 102, 241, 0.25), rgba(6, 182, 212, 0.08) 45%, transparent 80%);
  border-radius: 22px; border: 1px solid rgba(99, 102, 241, 0.3); backdrop-filter: blur(20px);
  box-shadow: 0 16px 40px -15px rgba(0, 0, 0, 0.7), 0 0 20px -5px rgba(99, 102, 241, 0.2);
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 8px; padding: 5px 16px;
  background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(6, 182, 212, 0.4);
  border-radius: 9999px; font-size: 0.82rem; font-weight: 600; color: #38bdf8;
  margin-bottom: 12px; letter-spacing: 0.03em; box-shadow: 0 0 12px rgba(6, 182, 212, 0.2);
}
.hero-title {
  font-size: 2.35rem !important; font-weight: 800 !important; letter-spacing: -0.035em !important;
  background: linear-gradient(135deg, #ffffff 15%, #bae6fd 55%, #c084fc 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 8px !important; line-height: 1.2 !important;
}
.hero-desc { font-size: 0.98rem !important; color: #94a3b8 !important; max-width: 660px; margin: 0 auto !important; line-height: 1.6; }
.chatbot-box {
  border-radius: 18px !important; border: 1px solid rgba(255, 255, 255, 0.08) !important;
  background: rgba(11, 16, 28, 0.75) !important; backdrop-filter: blur(18px) !important;
  box-shadow: 0 14px 40px -10px rgba(0, 0, 0, 0.6) !important;
}
.quick-chip {
  border-radius: 12px !important; font-size: 0.82rem !important; padding: 8px 14px !important;
  background: rgba(19, 27, 46, 0.8) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important;
  color: #cbd5e1 !important; font-weight: 500 !important;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.quick-chip:hover {
  background: rgba(99, 102, 241, 0.25) !important; border-color: rgba(6, 182, 212, 0.6) !important;
  color: #ffffff !important; box-shadow: 0 0 14px rgba(6, 182, 212, 0.3) !important; transform: translateY(-2px);
}
.send-btn {
  background: linear-gradient(135deg, #06b6d4 0%, #6366f1 50%, #9333ea 100%) !important;
  border: none !important; border-radius: 12px !important; color: #ffffff !important;
  font-weight: 700 !important; padding: 0 24px !important;
  box-shadow: 0 4px 18px rgba(99, 102, 241, 0.45) !important; transition: all 0.25s ease !important;
}
.send-btn:hover { box-shadow: 0 6px 24px rgba(6, 182, 212, 0.65) !important; transform: translateY(-2px) !important; }
.footer-text { text-align: center; font-size: 0.82rem; color: #64748b; margin-top: 28px; letter-spacing: 0.02em; }
"""


def chat_function(message, history):
    """Generate a RAG-powered response with sources and citations."""
    if rag_engine is None:
        return "RAG engine not initialized. Check GROQ_API_KEY secret."
    try:
        chunks = rag_engine.query(message, top_k=8)
        if not chunks:
            return (
                "I don't have enough information to answer that question. "
                "Please upload documents in the Document Ingestion tab to get started."
            )

        answer = rag_engine.generate_answer(message, chunks)

        sources_info = "\n\n---\n**Sources used:**\n"
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "unknown")
            page_info = f" (Page {chunk.get('page')})" if chunk.get("page") else ""
            score = chunk.get("score", None)
            score_str = f" `[RRF: {score:.4f}]`" if score is not None else ""
            sources_info += f"- **Source {i}:** `{source}`{page_info}{score_str}\n"

        return answer + sources_info

    except Exception as e:
        print(f"[ERROR] Error in chat function: {e}")
        return f"Error: {e}"


def get_index_info():
    """Get current index statistics."""
    if rag_engine is None:
        return "RAG engine not initialized. Check GROQ_API_KEY secret."
    stats = rag_engine.get_stats()
    if stats["total_chunks"] == 0:
        return "Index is empty. Upload documents to get started."

    sources_list = "\n".join(f" * {s}" for s in stats["sources"])
    return (
        f"**Index Statistics:**\n"
        f"- Total chunks: **{stats['total_chunks']}**\n"
        f"- Unique sources: **{stats['unique_sources']}**\n"
        f"- Active Documents:\n{sources_list}"
    )


def get_governance_dashboard():
    """Governance & trust layer summary: access roles, staleness, upload policies, and live audit log."""
    if rag_engine is None:
        return "RAG engine not initialized. Check GROQ_API_KEY secret."
    summary = rag_engine.get_audit_summary()

    role_lines = "\n".join(
        f" * **{r.upper()}**: {c} chunks" for r, c in summary["chunks_by_role"].items()
    ) or " * No documents indexed yet"

    log_lines = "\n".join(
        f" * `{time.strftime('%H:%M:%S', time.localtime(e['timestamp']))}` "
        f"\"{e['question'][:50]}\" → {', '.join(e['sources_retrieved'][:3]) or 'no match'}"
        for e in summary["recent_log"]
    ) or " * No queries logged yet"

    return (
        f"### 🛡️ Enterprise Governance & Trust Layer\n\n"
        f"**System Statistics:**\n"
        f"- Total queries logged: **{summary['total_queries']}**\n"
        f"- Stale chunks flagged (>180 days): **{summary['stale_chunk_count']}**\n\n"
        f"**Documents Indexed by Department / Role:**\n{role_lines}\n\n"
        f"**🔒 Departmental Upload Access Policies:**\n"
        f"- **Admin (Full Access)**: Authorized across all documents (Admin, HR, Finance, Public)\n"
        f"- **HR Department Head**: Restricted to HR & Public documents\n"
        f"- **Finance Department Head**: Restricted to Finance & Public documents\n"
        f"- **Staff / General User**: Restricted strictly to Public/General documents\n\n"
        f"**Recent Audit Log (last 15 queries):**\n{log_lines}"
    )


def clear_index():
    """Clear the entire RAG index."""
    if rag_engine is None:
        return "RAG engine not initialized. Check GROQ_API_KEY secret."
    rag_engine.clear_index()
    return "Index cleared. Upload new documents to rebuild the knowledge base."


def respond(message, chat_history):
    """Top-level respond function (must be picklable for ZeroGPU)."""
    if not message or not message.strip():
        return "", chat_history
    bot_message = chat_function(message, chat_history)
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})
    return "", chat_history


def clear_chat():
    """Top-level clear chat function (must be picklable for ZeroGPU)."""
    return []


# ════════════════════════════════════════════════════════
# BUILD GRADIO UI — demo must be a top-level variable
# ════════════════════════════════════════════════════════

demo = gr.Blocks(
    title="Universal RAG AI Knowledge Assistant",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
)

with demo:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")

    gr.HTML("""
    <div class="hero-banner">
      <div class="hero-badge">
        <span>Powered by Groq AI</span> | <span>Hybrid RRF Vector Search</span> | <span>Enterprise Governance</span>
      </div>
      <div class="hero-title">Universal RAG AI Assistant</div>
      <div class="hero-desc">
        Upload research papers, resumes, technical reports, or corporate files to build a
        context-grounded AI knowledge base with strict departmental upload governance.
      </div>
    </div>
    """)

    with gr.Tabs():
        with gr.TabItem("Chat", id="chat_tab"):
            chatbot = gr.Chatbot(
                label="Knowledge Assistant",
                height=480,
                elem_classes=["chatbot-box"],
            )

            gr.Markdown("###### Suggested Queries")
            with gr.Row():
                chip1 = gr.Button("Summarize all documents", elem_classes=["quick-chip"], size="sm")
                chip2 = gr.Button("Extract projects and skills", elem_classes=["quick-chip"], size="sm")
                chip3 = gr.Button("Explain methodology", elem_classes=["quick-chip"], size="sm")
                chip4 = gr.Button("Key metrics and benchmarks", elem_classes=["quick-chip"], size="sm")

            with gr.Row():
                msg = gr.Textbox(
                    label="Your Query",
                    placeholder="Ask anything about your documents...",
                    lines=2,
                    scale=5,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1, elem_classes=["send-btn"])

            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary", size="sm")

            chip1.click(lambda: "Summarize the key findings, candidates, and objectives across all uploaded documents.", outputs=msg)
            chip2.click(lambda: "What are the main projects, technical skills, and tools mentioned in the documents?", outputs=msg)
            chip3.click(lambda: "Explain the dataset, experimental methodology, and pipeline used in the research papers.", outputs=msg)
            chip4.click(lambda: "What are the key performance metrics, evaluation benchmarks, and quantitative results?", outputs=msg)

            submit_btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
            msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
            clear_btn.click(clear_chat, None, chatbot, queue=False)

        with gr.TabItem("Document Ingestion", id="ingest_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Enterprise Multi-Document Upload")
                    gr.Markdown("""
                    🔒 **Departmental Upload Access Control**:
                    * **Admin (Full Access)**: Authorized to upload all documents (Admin, HR, Finance, Public).
                    * **HR Department Head**: HR and Public documents only — **Finance & Admin blocked**.
                    * **Finance Department Head**: Finance and Public documents only — **HR & Admin blocked**.
                    * **Staff / General User**: Public/General documents only — **all sensitive docs blocked**.

                    > ⚠️ You **must** declare the Document Type before uploading. The system enforces this — no bypass via filename.
                    """)
                    uploader_role_dropdown = gr.Dropdown(
                        UPLOAD_ROLES,
                        value="Admin (Full Access)",
                        label="🔐 Your Role / Department",
                    )
                    doc_type_dropdown = gr.Dropdown(
                        DOCUMENT_TYPES,
                        value="Public / General Document",
                        label="📂 Document Type (declare before uploading)",
                    )
                    file_input = gr.File(
                        label="Select Files (PDF, DOCX, TXT, CSV, XLSX, XLS)",
                        file_types=[".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls"],
                        file_count="multiple",
                    )
                    ingest_file_btn = gr.Button("Ingest Document(s)", variant="primary", elem_classes=["send-btn"])
                    file_status = gr.Textbox(label="Ingestion & Security Log", interactive=False, lines=8)

                with gr.Column(scale=1):
                    gr.Markdown("### Index Management")
                    gr.Markdown("Check indexed chunks or clear the vector store to upload new documents.")
                    index_info = gr.Textbox(label="Vector Store Status", interactive=False, lines=6)
                    with gr.Row():
                        refresh_btn = gr.Button("Refresh Stats", variant="secondary")
                        clear_index_btn = gr.Button("Clear Index", variant="stop")

            ingest_file_btn.click(ingest_file, inputs=[file_input, uploader_role_dropdown, doc_type_dropdown], outputs=file_status)
            refresh_btn.click(get_index_info, outputs=index_info)
            clear_index_btn.click(clear_index, outputs=index_info)

        with gr.TabItem("Governance Dashboard", id="governance_tab"):
            gr.Markdown("### 🛡️ Enterprise Governance & Trust Layer")
            gr.Markdown("Real-time monitoring of document departmental distribution, staleness, upload policies, and query audit trails.")
            dashboard_output = gr.Markdown("Click 'Refresh Dashboard' to load live governance statistics.")
            refresh_dash_btn = gr.Button("Refresh Dashboard", variant="primary", elem_classes=["send-btn"])
            refresh_dash_btn.click(get_governance_dashboard, outputs=dashboard_output)

        with gr.TabItem("Architecture", id="about_tab"):
            gr.Markdown("""
            ### Technical Architecture and Pipeline

            ```
            [Uploaded Files: PDF / DOCX / TXT / CSV / XLSX]  --(Role-Validated by Department Head)-->
                    │
                    ▼
            [Page & Sheet-Aware Chunking (pypdf + pandas/openpyxl + RecursiveSplitter)]
                    │
                    ▼
            [SentenceTransformers: all-MiniLM-L6-v2 (384-dim)]
                    │
                    ▼
            [FAISS Vector Store FlatL2] + [Source-Aware Keyword Matcher]
                    │
                    ▼
            [Reciprocal Rank Fusion (RRF): Score = 1/(60+R_sem) + 1/(60+R_kw)]
                    │
                    ▼
            [Governance Layer: Department Upload RBAC + Staleness Flag + Live Audit Log]
                    │
                    ▼
            [Groq Qwen 3.6 27B LLM Inference with Structured Prompting and Citations]
                    │
                    ▼
            [Direct Polished User Answer with Source, Sheet, and Page Citations]
            ```

            | Layer | Component | Detail |
            | :--- | :--- | :--- |
            | **Upload Security** | Department Head Validation | Enforces upload authorization (Admin / HR / Finance / Staff) |
            | **LLM Inference** | Groq API (`qwen/qwen3.6-27b`) | Ultra-fast token generation and deep synthesis |
            | **Vector Database** | FAISS | In-memory L2 similarity search |
            | **Embeddings** | `all-MiniLM-L6-v2` | Dense sentence representations |
            | **Hybrid Ranker** | Reciprocal Rank Fusion (RRF) | Balances semantic similarity and keyword precision |
            | **Governance** | Trust & Audit Layer | Real-time query audit trail & staleness tracking (>180d) |
            | **Parsing** | Page-Aware Native pypdf | Preserves page numbers and table context |
            """)

    gr.HTML('<div class="footer-text">Universal RAG AI Knowledge Assistant | Powered by Groq and FAISS</div>')


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    is_cloud = server_name == "0.0.0.0"
    demo.launch(ssr_mode=False, server_name=server_name, server_port=port, share=not is_cloud)
else:
    pass
