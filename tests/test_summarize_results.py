import pandas as pd

from scripts.summarize_results import summarize_results


def test_summarize_results_writes_at_least_ten_pdf_graphs(tmp_path):
    rows = [
        {
            "plan_id": "lora_rank_8",
            "tradefl_score": 0.4,
            "peak_memory_bytes": 10,
            "compute_to_target_seconds": 2,
            "communication_to_target_bytes": 30,
            "latency_to_target_seconds": 5,
            "validation_utility": 0.85,
            "accuracy_loss": 0.05,
            "privacy_risk": 0.2,
            "rounds_to_target": 2,
            "feasible": True,
        },
        {
            "plan_id": "qlora_4bit_rank_8",
            "tradefl_score": 0.3,
            "peak_memory_bytes": 5,
            "compute_to_target_seconds": 3,
            "communication_to_target_bytes": 20,
            "latency_to_target_seconds": 6,
            "validation_utility": 0.84,
            "accuracy_loss": 0.06,
            "privacy_risk": 0.1,
            "rounds_to_target": 3,
            "feasible": True,
        },
    ]
    input_path = tmp_path / "plan_summary.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    output_prefix = tmp_path / "summary"

    summarize_results(input_path, output_prefix)

    assert (tmp_path / "summary.csv").exists()
    pdfs = sorted((tmp_path / "summary_graphs").glob("*.pdf"))
    assert len(pdfs) >= 10
    assert all(path.stat().st_size > 0 for path in pdfs)
    assert (tmp_path / "summary_graphs" / "graph_manifest.csv").exists()
