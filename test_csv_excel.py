import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_engine import RagEngine

def test_csv_and_excel():
    print("--- Testing CSV and Excel Ingestion & Search ---")
    engine = RagEngine()

    # Create sample CSV
    csv_path = os.path.join(os.path.dirname(__file__), "sample_docs", "employees.csv")
    df_csv = pd.DataFrame({
        "EmployeeID": [101, 102, 103],
        "Name": ["Alice Johnson", "Bob Smith", "Charlie Davis"],
        "Department": ["Engineering", "HR", "Finance"],
        "Salary": [95000, 72000, 88000],
        "Role": ["Senior AI Engineer", "HR Specialist", "Financial Analyst"]
    })
    df_csv.to_csv(csv_path, index=False)
    print(f"[OK] Created sample CSV: {csv_path}")

    # Create sample Excel (.xlsx)
    xlsx_path = os.path.join(os.path.dirname(__file__), "sample_docs", "quarterly_budget.xlsx")
    df_xlsx = pd.DataFrame({
        "Quarter": ["Q1", "Q2", "Q3", "Q4"],
        "Revenue_M": [10.5, 12.0, 14.2, 16.5],
        "EBITDA_Margin": ["25%", "27%", "28.5%", "30%"],
        "Key_Initiative": ["Cloud Migration", "AI Cluster Deployment", "Enterprise RAG Engine", "Global Expansion"]
    })
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_xlsx.to_excel(writer, sheet_name="Financial_Summary", index=False)
    print(f"[OK] Created sample Excel: {xlsx_path}")

    # Ingest CSV
    engine.ingest_document(csv_path, doc_role="public")
    print("[OK] Ingested employees.csv successfully")

    # Ingest Excel
    engine.ingest_document(xlsx_path, doc_role="finance")
    print("[OK] Ingested quarterly_budget.xlsx successfully")

    # Verify CSV query
    res_csv = engine.query("Who is the Senior AI Engineer and what is their salary?", top_k=5)
    assert len(res_csv) > 0, "Expected search results for CSV query"
    print(f"[OK] CSV search matched source: {res_csv[0]['source']} (Score: {res_csv[0]['score']:.4f})")

    # Verify Excel query
    res_xlsx = engine.query("What was Q3 revenue and EBITDA margin?", top_k=5)
    assert len(res_xlsx) > 0, "Expected search results for Excel query"
    print(f"[OK] Excel search matched source: {res_xlsx[0]['source']} (Score: {res_xlsx[0]['score']:.4f})")

    # Test Answer Generation
    answer = engine.generate_answer("Who is the Senior AI Engineer and what is their salary?", res_csv)
    print(f"[OK] Generated Answer:\n{answer}")

    print("--- ALL CSV AND EXCEL TESTS PASSED SUCCESSFULLY! ---")

if __name__ == "__main__":
    test_csv_and_excel()
