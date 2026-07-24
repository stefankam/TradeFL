#!/usr/bin/env python
"""Obtain and prepare MedNLI/PubMedQA inputs for TradeFL.

PubMedQA can be downloaded from the public PubMedQA GitHub repository. MedNLI is
credentialed PhysioNet data; this script prepares/validates MedNLI only after a
credentialed user places the licensed files locally.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PUBMEDQA_ORI_PQAL_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
PUBMEDQA_GROUND_TRUTH_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/prepare TradeFL datasets.")
    parser.add_argument("--dataset", choices=["pubmedqa", "mednli", "all"], default="all")
    parser.add_argument("--pubmedqa-dir", default="data/PubMedQA")
    parser.add_argument("--mednli-dir", default="data/mednli")
    parser.add_argument("--mednli-source-dir", help="Directory containing licensed MedNLI train/dev/test JSONL files downloaded from PhysioNet.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset in {"pubmedqa", "all"}:
        obtain_pubmedqa(Path(args.pubmedqa_dir), args.seed)
    if args.dataset in {"mednli", "all"}:
        prepare_mednli(Path(args.mednli_dir), Path(args.mednli_source_dir) if args.mednli_source_dir else None)


def ensure_dataset_available(dataset_config: dict[str, Any], seed: int = 42) -> None:
    """Ensure configured data files exist before the dataloader is called.

    PubMedQA is public, so missing configured PubMedQA splits trigger a download
    from the PubMedQA GitHub data source. MedNLI is credentialed, so missing
    configured MedNLI splits create an access note and then fail with a clear
    instruction to provide the licensed files.
    """

    name = str(dataset_config.get("name", "")).lower()
    paths = dataset_config.get("paths", {})
    missing = [Path(str(path)) for path in paths.values() if path and not Path(str(path)).exists()]
    if not missing:
        return
    if name == "pubmedqa":
        output_dir = _dataset_output_dir(paths, default=Path("data/PubMedQA"))
        obtain_pubmedqa(output_dir, seed)
        return
    if name == "mednli":
        output_dir = _dataset_output_dir(paths, default=Path("data/mednli"))
        prepare_mednli(output_dir, None)
        raise FileNotFoundError(
            "MedNLI files are missing and cannot be auto-downloaded because MedNLI "
            "requires PhysioNet credentialing and a signed DUA. See "
            f"{output_dir / 'MEDNLI_REQUIRES_PHYSIONET_ACCESS.txt'}."
        )
    raise ValueError("dataset.name must be either 'mednli' or 'pubmedqa'.")


def _dataset_output_dir(paths: dict[str, Any], default: Path) -> Path:
    for key in ("train", "validation", "test"):
        value = paths.get(key)
        if value:
            return Path(str(value)).parent
    return default


def obtain_pubmedqa(output_dir: Path, seed: int) -> None:
    """Download PubMedQA PQA-L and write train/validation/test JSONL files."""

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ori_path = raw_dir / "ori_pqal.json"
    truth_path = raw_dir / "test_ground_truth.json"
    _download(PUBMEDQA_ORI_PQAL_URL, ori_path)
    _download(PUBMEDQA_GROUND_TRUTH_URL, truth_path)
    with ori_path.open(encoding="utf-8") as handle:
        data: dict[str, dict[str, Any]] = json.load(handle)
    items = sorted(data.items())
    random.Random(seed).shuffle(items)
    train_end = int(0.8 * len(items))
    validation_end = int(0.9 * len(items))
    splits = {
        "train": items[:train_end],
        "validation": items[train_end:validation_end],
        "test": items[validation_end:],
    }
    for split, rows in splits.items():
        _write_pubmedqa_jsonl(output_dir / f"{split}.jsonl", rows)
    _write_manifest(
        output_dir / "MANIFEST.json",
        {
            "dataset": "PubMedQA",
            "source": "pubmedqa/pubmedqa GitHub repository",
            "source_urls": [PUBMEDQA_ORI_PQAL_URL, PUBMEDQA_GROUND_TRUTH_URL],
            "split_strategy": "deterministic 80/10/10 shuffle over PQA-L using --seed",
            "seed": seed,
            "records": {split: len(rows) for split, rows in splits.items()},
        },
    )


def prepare_mednli(output_dir: Path, source_dir: Path | None) -> None:
    """Copy licensed MedNLI files from a local PhysioNet download into data/mednli."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if source_dir is None:
        (output_dir / "MEDNLI_REQUIRES_PHYSIONET_ACCESS.txt").write_text(
            "MedNLI is credentialed PhysioNet data. Download it from PhysioNet after "
            "credentialing and signing the DUA, then run:\n\n"
            "python scripts/obtain_data.py --dataset mednli --mednli-source-dir /path/to/mednli\n",
            encoding="utf-8",
        )
        return
    candidates = {
        "train": ["train.jsonl", "mli_train_v1.jsonl", "mednli_train.jsonl"],
        "validation": ["validation.jsonl", "dev.jsonl", "mli_dev_v1.jsonl", "mednli_dev.jsonl"],
        "test": ["test.jsonl", "mli_test_v1.jsonl", "mednli_test.jsonl"],
    }
    copied: dict[str, str] = {}
    for split, names in candidates.items():
        source = _find_first(source_dir, names)
        if source is None:
            raise FileNotFoundError(f"Could not find a MedNLI {split} split in {source_dir}; tried {names}.")
        destination = output_dir / f"{split}.jsonl"
        shutil.copyfile(source, destination)
        copied[split] = str(source)
    _write_manifest(
        output_dir / "MANIFEST.json",
        {
            "dataset": "MedNLI",
            "source": "PhysioNet credentialed download supplied by the user",
            "copied_from": copied,
            "note": "MedNLI files are not redistributed by this repository.",
        },
    )


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "TradeFL-data-obtainer/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            destination.write_bytes(response.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not download {url}. If this environment blocks GitHub/PhysioNet, "
            "run this script on a network with dataset access, or manually place the "
            f"source file at {destination}. Original error: {exc}"
        ) from exc


def _write_pubmedqa_jsonl(path: Path, rows: list[tuple[str, dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for pmid, row in rows:
            normalized = {
                "pmid": pmid,
                "question": row.get("QUESTION", ""),
                "context": row.get("CONTEXTS", []),
                "labels": row.get("LABELS", []),
                "final_decision": row.get("final_decision", row.get("reasoning_required_pred", "")),
                "long_answer": row.get("LONG_ANSWER", ""),
            }
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")


def _find_first(directory: Path, names: list[str]) -> Path | None:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
