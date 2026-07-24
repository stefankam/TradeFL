import json

from scripts.obtain_data import obtain_pubmedqa, prepare_mednli


def test_prepare_mednli_without_source_writes_access_note(tmp_path):
    prepare_mednli(tmp_path, None)
    note = tmp_path / "MEDNLI_REQUIRES_PHYSIONET_ACCESS.txt"
    assert note.exists()
    assert "PhysioNet" in note.read_text(encoding="utf-8")


def test_obtain_pubmedqa_uses_existing_raw_files(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = {
        str(index): {
            "QUESTION": f"Question {index}?",
            "CONTEXTS": [f"Context {index}"],
            "LABELS": ["BACKGROUND"],
            "final_decision": "yes" if index % 2 else "no",
            "LONG_ANSWER": "answer",
        }
        for index in range(10)
    }
    (raw / "ori_pqal.json").write_text(json.dumps(rows), encoding="utf-8")
    (raw / "test_ground_truth.json").write_text("{}", encoding="utf-8")
    obtain_pubmedqa(tmp_path, seed=7)
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "validation.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["records"] == {"train": 8, "validation": 1, "test": 1}


def test_ensure_dataset_available_downloads_pubmedqa_when_splits_missing(tmp_path):
    from scripts import obtain_data

    output_dir = tmp_path / "PubMedQA"
    calls = []

    def fake_obtain_pubmedqa(path, seed):
        calls.append((path, seed))
        for split in ["train", "validation", "test"]:
            (path / f"{split}.jsonl").parent.mkdir(parents=True, exist_ok=True)
            (path / f"{split}.jsonl").write_text("{}\n", encoding="utf-8")

    original = obtain_data.obtain_pubmedqa
    obtain_data.obtain_pubmedqa = fake_obtain_pubmedqa
    try:
        obtain_data.ensure_dataset_available(
            {
                "name": "pubmedqa",
                "paths": {
                    "train": output_dir / "train.jsonl",
                    "validation": output_dir / "validation.jsonl",
                    "test": output_dir / "test.jsonl",
                },
            },
            seed=123,
        )
    finally:
        obtain_data.obtain_pubmedqa = original
    assert calls == [(output_dir, 123)]
