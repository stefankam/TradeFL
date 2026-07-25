import pandas as pd
import pytest
from scripts.summarize_results import summarize_results


def test_summarize_results_writes_at_least_ten_pdf_graphs(tmp_path):
    rows = [
        {
            "plan_id": "lora_rank_8",
            "training_mode": "real_federated",
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
            "training_mode": "real_federated",
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


def test_real_federated_graphs_require_matching_provenance(tmp_path):
    input_path = tmp_path / "plan_summary.csv"
    pd.DataFrame([{"plan_id": "reference", "validation_utility": 0.5}]).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="training_mode='real_federated'"):
        summarize_results(input_path, tmp_path / "summary")


def test_real_federated_provenance_allows_graph_generation(tmp_path):
    input_path = tmp_path / "plan_summary.csv"
    pd.DataFrame(
        [{"plan_id": "llama3_8b_lora", "training_mode": "real_federated", "validation_utility": 0.8}]
    ).to_csv(input_path, index=False)

    summarize_results(input_path, tmp_path / "summary")

    assert len(list((tmp_path / "summary_graphs").glob("*.pdf"))) == 12
