#!/usr/bin/env python
"""Run sequential, real Transformer client training with sample-weighted FedAvg."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.obtain_data import ensure_dataset_available
from tradefl.data import load_dataset_bundle
from tradefl.federation.fedavg import iid_partition_indices, sample_weighted_fedavg
from tradefl.federation.huggingface import (
    HuggingFaceClientTrainer,
    UnsupportedRuntimeError,
    tensor_state_nbytes,
    unavailable_runtime_reason,
)
from tradefl.utils.config import load_yaml
from tradefl.utils.seeds import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", default="configs/experiment_pubmedqa.yaml")
    parser.add_argument("--models-config", default="configs/real_federated_models.yaml")
    parser.add_argument("--output-dir", default="outputs/real_federated")
    parser.add_argument("--experiment-id", action="append", help="Run only the selected experiment ID; repeatable.")
    parser.add_argument("--strict-hardware", action="store_true", help="Fail instead of skipping models unsupported by this host.")
    parser.add_argument(
        "--cpu-smoke-test",
        action="store_true",
        help="Run a short two-client, one-round, one-seed Bio_ClinicalBERT pipeline check on CPU.",
    )
    parser.add_argument(
        "--allow-slow-cpu",
        action="store_true",
        help="Allow the full Bio_ClinicalBERT study on CPU; it can take many hours or days.",
    )
    args = parser.parse_args()

    experiment_cfg = load_yaml(args.experiment_config)
    models_cfg = load_yaml(args.models_config)
    exp = experiment_cfg["experiment"]
    training = {**models_cfg.get("training", {}), **{"local_epochs": exp.get("local_epochs", 1)}}
    if args.cpu_smoke_test:
        experiment_cfg, training = apply_cpu_smoke_profile(experiment_cfg, training)
        exp = experiment_cfg["experiment"]
    training["allow_slow_cpu"] = bool(args.allow_slow_cpu or args.cpu_smoke_test)
    ensure_dataset_available(experiment_cfg["dataset"], seed=int(exp["seeds"][0]))
    dataset = load_dataset_bundle(experiment_cfg["dataset"])
    selected = set(args.experiment_id or [])
    experiments = [
        item for item in models_cfg["federated_experiments"] if not selected or item["experiment_id"] in selected
    ]
    if args.cpu_smoke_test:
        experiments = [item for item in experiments if item["experiment_id"] == "bio_clinicalbert_baseline"]
    if selected - {item["experiment_id"] for item in experiments}:
        raise ValueError(f"Unknown experiment IDs: {sorted(selected - {item['experiment_id'] for item in experiments})}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    round_path = output / "round_metrics.jsonl"
    round_path.unlink(missing_ok=True)
    summaries = []
    skipped = []
    for model_config in experiments:
        unavailable = unavailable_runtime_reason(model_config, training)
        if unavailable:
            if args.strict_hardware:
                raise SystemExit(f"UNSUPPORTED {model_config['experiment_id']}: {unavailable}")
            skipped.append({"experiment_id": model_config["experiment_id"], "seed": None, "reason": unavailable})
            print(f"SKIPPED {model_config['experiment_id']}: {unavailable}", file=sys.stderr)
            continue
        for seed in exp["seeds"]:
            try:
                summaries.append(run_model_experiment(model_config, training, exp, dataset, seed, round_path))
            except UnsupportedRuntimeError as exc:
                if args.strict_hardware:
                    raise
                skipped.append({"experiment_id": model_config["experiment_id"], "seed": seed, "reason": str(exc)})
                print(f"SKIPPED {model_config['experiment_id']} seed={seed}: {exc}", file=sys.stderr)
                break
    (output / "skipped_experiments.json").write_text(json.dumps(skipped, indent=2) + "\n", encoding="utf-8")
    if not summaries:
        raise SystemExit(
            "No selected experiment can run with these hardware settings. On CPU use --cpu-smoke-test; "
            "for the full study use CUDA (or explicitly acknowledge a very slow BERT run with --allow-slow-cpu)."
        )
    pd.DataFrame(summaries).to_csv(output / "raw_metrics.csv", index=False)


def apply_cpu_smoke_profile(experiment_cfg: dict, training: dict) -> tuple[dict, dict]:
    """Create an explicit, practical CPU profile without changing the full study."""

    experiment_cfg = copy.deepcopy(experiment_cfg)
    experiment_cfg["experiment"].update(
        {
            "num_clients": 2,
            "client_sampling_ratio": 1.0,
            "max_rounds": 1,
            "seeds": [experiment_cfg["experiment"]["seeds"][0]],
        }
    )
    training = {
        **training,
        "use_cpu": True,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_length": 128,
        "evaluation_batch_size": 8,
        "max_train_samples_per_client": 8,
        "cpu_threads": 4,
        "local_epochs": 1,
    }
    return experiment_cfg, training


def run_model_experiment(model_config, training, exp, dataset, seed, round_path: Path) -> dict:
    """Run all federated rounds for one base-model architecture and seed."""

    set_seed(seed)
    num_clients = int(exp["num_clients"])
    partitions = iid_partition_indices(len(dataset.train), num_clients, seed)
    client_records = [[dataset.train[int(index)] for index in partition] for partition in partitions]
    trainer = HuggingFaceClientTrainer(model_config, dataset.labels, training)
    global_state = trainer.initial_state()
    rounds = []
    target_reached = False
    rounds_to_target = None
    rng = np.random.default_rng(seed)
    clients_per_round = max(2, math.ceil(num_clients * float(exp.get("client_sampling_ratio", 1.0))))
    clients_per_round = min(num_clients, clients_per_round)

    for round_index in range(int(exp["max_rounds"])):
        selected_clients = sorted(rng.choice(num_clients, size=clients_per_round, replace=False).tolist())
        downloads = tensor_state_nbytes(global_state) * len(selected_clients)
        updates = []
        compute_seconds = 0.0
        peak_memory = 0
        uploaded = 0
        round_started = time.perf_counter()
        for client_id in selected_clients:
            local_records = client_records[client_id]
            sample_limit = training.get("max_train_samples_per_client")
            if sample_limit is not None:
                local_records = local_records[: int(sample_limit)]
            print(
                f"TRAINING {model_config['experiment_id']} seed={seed} "
                f"round={round_index + 1}/{exp['max_rounds']} "
                f"client={client_id + 1}/{num_clients} examples={len(local_records)}",
                flush=True,
            )
            update = trainer.train_client(local_records, global_state)
            updates.append((update.state, update.num_examples))
            compute_seconds += update.compute_seconds
            peak_memory = max(peak_memory, update.peak_accelerator_memory_bytes)
            uploaded += update.uploaded_bytes
            downloads += update.downloaded_bytes
            print(f"COMPLETED client={client_id + 1}/{num_clients} in {update.compute_seconds:.1f}s", flush=True)
        global_state = sample_weighted_fedavg(updates)
        validation = trainer.evaluate(dataset.validation, global_state)
        test = trainer.evaluate(dataset.test, global_state)
        latency = time.perf_counter() - round_started
        row = {
            "plan_id": model_config["experiment_id"],
            "model_id": model_config["model_id"],
            "method": model_config.get("method", "full_finetuning"),
            "seed": seed,
            "round_index": round_index,
            "selected_clients": selected_clients,
            "client_example_counts": [len(client_records[index]) for index in selected_clients],
            "peak_memory_bytes": peak_memory,
            "compute_time_seconds": compute_seconds,
            "bytes_uploaded": uploaded,
            "bytes_downloaded": downloads,
            "latency_seconds": latency,
            "validation_utility": validation[exp.get("primary_metric", "accuracy")],
            "validation_macro_f1": validation["macro_f1"],
            "test_utility": test[exp.get("primary_metric", "accuracy")],
            "test_macro_f1": test["macro_f1"],
            "privacy_risk": float(model_config.get("privacy_risk", 0.25)),
            "training_mode": "real_federated",
            "aggregation": "FedAvg",
        }
        with round_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        rounds.append(row)
        if row["validation_utility"] >= float(exp["target_quality"]):
            target_reached = True
            rounds_to_target = round_index + 1
            break

    final = rounds[-1]
    return {
        "plan_id": model_config["experiment_id"],
        "model_id": model_config["model_id"],
        "method": model_config.get("method", "full_finetuning"),
        "seed": seed,
        "dataset": dataset.name,
        "training_mode": "real_federated",
        "aggregation": "FedAvg",
        "validation_utility": final["validation_utility"],
        "validation_macro_f1": final["validation_macro_f1"],
        "test_utility": final["test_utility"],
        "test_macro_f1": final["test_macro_f1"],
        "accuracy_loss": max(0.0, float(exp.get("reference_utility", 1.0)) - final["validation_utility"]),
        "rounds_completed": len(rounds),
        "rounds_to_target": rounds_to_target,
        "target_reached": target_reached,
        "peak_memory_bytes": max(row["peak_memory_bytes"] for row in rounds),
        "compute_to_target_seconds": sum(row["compute_time_seconds"] for row in rounds),
        "communication_to_target_bytes": sum(row["bytes_uploaded"] + row["bytes_downloaded"] for row in rounds),
        "latency_to_target_seconds": sum(row["latency_seconds"] for row in rounds),
        "mean_round_latency_seconds": sum(row["latency_seconds"] for row in rounds) / len(rounds),
        "privacy_risk": final["privacy_risk"],
        "energy_to_target_joules": None,
    }


if __name__ == "__main__":
    main()
