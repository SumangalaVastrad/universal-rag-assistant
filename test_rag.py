import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_engine import RagEngine

def run_tests():
    print("--- Starting RagEngine Automated Verification ---")
    engine = RagEngine()
    print("[OK] RagEngine initialized successfully")

    # Ingest public doc
    pub_doc = os.path.join(os.path.dirname(__file__), "sample_docs", "research_paper.txt")
    engine.ingest_document(pub_doc, doc_role="public")
    print("[OK] Ingested research_paper.txt as 'public'")

    # Ingest finance doc
    fin_doc = os.path.join(os.path.dirname(__file__), "sample_docs", "finance_report.txt")
    engine.ingest_document(fin_doc, doc_role="finance")
    print("[OK] Ingested finance_report.txt as 'finance'")

    stats = engine.get_stats()
    print(f"[OK] Stats check: {stats}")
    assert stats["total_chunks"] >= 2, "Expected at least 2 chunks"
    assert stats["unique_sources"] == 2, "Expected 2 unique sources"

    # Test 1: Public user querying finance doc -> should return 0 finance chunks
    res_public = engine.query("What was Q3 revenue and EBITDA?", top_k=5, user_role="public")
    for r in res_public:
        assert r["doc_role"] == "public", f"Public user got forbidden doc_role: {r['doc_role']}"
    print("[OK] Governance test passed: 'public' role cannot access 'finance' docs")

    # Test 2: Finance user querying finance doc -> should return finance chunk
    res_finance = engine.query("What was Q3 revenue and EBITDA?", top_k=5, user_role="finance")
    assert len(res_finance) > 0, "Finance user should find matching documents"
    finance_sources = [r["source"] for r in res_finance]
    assert "finance_report.txt" in finance_sources, "Expected finance_report.txt in results"
    print(f"[OK] Finance query succeeded with RRF score: {res_finance[0]['score']:.4f}")

    # Test 3: Public query on research paper
    res_pub_query = engine.query("What is the retrieval latency and Recall@8?", top_k=5, user_role="public")
    assert len(res_pub_query) > 0
    answer = engine.generate_answer("What is the retrieval latency and Recall@8?", res_pub_query)
    assert len(answer) > 0
    print("[OK] Answer generation succeeded")

    # Test 4: Audit summary
    summary = engine.get_audit_summary()
    print(f"[OK] Governance summary: total_queries={summary['total_queries']}, roles={summary['queries_by_role']}")
    assert summary["total_queries"] >= 3

    print("--- ALL RAG ENGINE TESTS PASSED! ---")

if __name__ == "__main__":
    run_tests()
