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
from collections.abc import Callable
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.obtain_data import ensure_dataset_available
from tradefl.data import load_dataset_bundle
from tradefl.backends.openai_baseline import evaluate_openai_baseline
from tradefl.backends.huggingface_baseline import evaluate_huggingface_baseline
from tradefl.federation.fedavg import iid_partition_indices, sample_weighted_fedavg
from tradefl.federation.huggingface import (
    HuggingFaceClientTrainer,
    UnsupportedRuntimeError,
    tensor_state_nbytes,
    unavailable_runtime_reason,
)
from tradefl.measurement.energy import EnergyMeter
from tradefl.scheduling import ONLINE_POLICIES, ShadowPriceScheduler, build_online_scheduler
from tradefl.selection.feasibility import Constraints
from tradefl.selection.normalization import Budgets
from tradefl.selection.scoring import TradeFLWeights
from tradefl.selection.selector import select_best, write_selection
from tradefl.utils.config import load_yaml
from tradefl.utils.seeds import set_seed
from scripts.summarize_results import summarize_results
from scripts.select_plan import entropy_weights



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", default="configs/experiment_pubmedqa.yaml")
    parser.add_argument("--models-config", default="configs/real_federated_models.yaml")
    parser.add_argument("--output-dir", default="outputs/real_federated")
    parser.add_argument("--budgets", default="configs/budgets.yaml")
    parser.add_argument("--weights", default="configs/weights.yaml")
    parser.add_argument("--weight-method", choices=["configured", "entropy"], default="configured")
    parser.add_argument("--experiment-id", action="append", help="Run only the selected experiment ID; repeatable.")
    parser.add_argument(
        "--scheduler-policy", action="append", choices=ONLINE_POLICIES,
        help="Run only this online scheduling treatment; repeatable.",
    )
    parser.add_argument("--strict-hardware", action="store_true", help="Fail instead of skipping models unsupported by this host.")
    parser.add_argument(
        "--cpu-smoke-test",
        action="store_true",
        help="Run a short two-client, one-round, one-seed Bio_ClinicalBERT pipeline check on CPU.",
    )

    parser.add_argument(
        "--run-external-baselines",
        action="store_true",
        help="Evaluate configured centralized API baselines (may send dataset records to paid external APIs).",
    )
    parser.add_argument(
        "--external-only",
        action="store_true",
        help="Skip every federated experiment and evaluate only configured centralized API baselines.",
    )
    parser.add_argument(
        "--external-baseline-id",
        action="append",
        help="Run only the selected external baseline ID; repeatable and implies external baseline execution.",
    )
    parser.add_argument(
        "--append-results",
        action="store_true",
        help="Merge an external-only result into an existing raw_metrics.csv without deleting federated round logs.",
    )
    parser.add_argument(
        "--allow-slow-cpu",
        action="store_true",
        help="Allow the full Bio_ClinicalBERT study on CPU; it can take many hours or days.",
    )

    args = parser.parse_args()

    if args.external_only and args.cpu_smoke_test:
        parser.error("--external-only cannot be combined with --cpu-smoke-test")
    if args.external_only and args.experiment_id:
        parser.error("--external-only cannot be combined with --experiment-id")

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
    if args.external_only:
        experiments = []
    if args.cpu_smoke_test:
        experiments = [item for item in experiments if item["experiment_id"] == "bio_clinicalbert_baseline"]
    if selected - {item["experiment_id"] for item in experiments}:
        raise ValueError(f"Unknown experiment IDs: {sorted(selected - {item['experiment_id'] for item in experiments})}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    round_path = output / "round_metrics.jsonl"

    raw_path = output / "raw_metrics.csv"
    if args.append_results and not raw_path.exists():
        raise SystemExit(f"Cannot append results because {raw_path} does not exist.")
    if not args.append_results:
        round_path.unlink(missing_ok=True)
    summaries = pd.read_csv(raw_path).to_dict("records") if args.append_results else []
    skipped_path = output / "skipped_experiments.json"
    skipped = json.loads(skipped_path.read_text(encoding="utf-8")) if args.append_results and skipped_path.exists() else []
    def publish(partial: dict | None = None) -> None:
        rows = [*summaries, *([] if partial is None else [partial])]
        if rows:
            write_live_reports(rows, output, args, experiment_cfg, models_cfg)


    for model_config in experiments:
        unavailable = unavailable_runtime_reason(model_config, training)
        if unavailable:
            if args.strict_hardware:
                raise SystemExit(f"UNSUPPORTED {model_config['experiment_id']}: {unavailable}")
            skipped.append({"experiment_id": model_config["experiment_id"], "seed": None, "reason": unavailable})
            print(f"SKIPPED {model_config['experiment_id']}: {unavailable}", file=sys.stderr)
            continue

        treatments = scheduler_treatments(exp, args.scheduler_policy)
        for policy in treatments:
            treatment_config = copy.deepcopy(model_config)
            treatment_config["base_experiment_id"] = model_config["experiment_id"]
            treatment_config["scheduler_policy"] = policy
            treatment_config["experiment_id"] = (
                f"{model_config['experiment_id']}__{policy}" if policy is not None else model_config["experiment_id"]
            )
            for seed in exp["seeds"]:
                try:
                    summaries.append(
                        run_model_experiment(
                            treatment_config,
                            training,
                            exp,
                            dataset,
                            seed,
                            round_path,
                            on_round=publish,
                            scheduling_policy=policy,
                            constraints=experiment_cfg.get("constraints", {}),
                        )
                    )
                except UnsupportedRuntimeError as exc:
                    if args.strict_hardware:
                        raise
                    skipped.append({"experiment_id": treatment_config["experiment_id"], "seed": seed, "reason": str(exc)})
                    print(f"SKIPPED {treatment_config['experiment_id']} seed={seed}: {exc}", file=sys.stderr)
                    break
    if args.run_external_baselines or args.external_only or args.external_baseline_id:
        external_models = models_cfg.get("external_models", [])
        selected_external = set(args.external_baseline_id or [])
        known_external = {str(item.get("experiment_id", item["model_id"])) for item in external_models}
        if selected_external - known_external:
            raise ValueError(f"Unknown external baseline IDs: {sorted(selected_external - known_external)}")
        for model_config in external_models:
            external_id = str(model_config.get("experiment_id", model_config["model_id"]))
            if selected_external and external_id not in selected_external:
                continue
            if model_config.get("method") == "centralized_api":
                row = evaluate_openai_baseline(
                    model_config,
                    dataset.validation,
                    dataset.test,
                    dataset.labels,
                )
            elif model_config.get("method") == "centralized_huggingface":
                row = evaluate_huggingface_baseline(model_config, dataset, training)
            else:
                continue
            row["accuracy_loss"] = max(
                0.0,
                float(exp.get("reference_utility", 1.0)) - float(row["validation_utility"]),
            )
            row["target_reached"] = row["validation_utility"] >= float(exp["target_quality"])
            row["target_quality"] = float(exp["target_quality"])
            row["constraint_provenance"] = json.dumps(experiment_cfg.get("constraints", {}), sort_keys=True)
            row["scheduler_policy"] = "centralized_baseline"
            summaries[:] = [existing for existing in summaries if str(existing.get("plan_id")) != external_id]
            summaries.append(row)
            publish()
    skipped_path.write_text(json.dumps(skipped, indent=2) + "\n", encoding="utf-8")
    if not summaries:
        raise SystemExit(
            "No selected experiment can run with these hardware settings. On CPU use --cpu-smoke-test; "
            "for the full study use CUDA (or explicitly acknowledge a very slow BERT run with --allow-slow-cpu)."
        )
    publish()




def apply_cpu_smoke_profile(experiment_cfg: dict, training: dict) -> tuple[dict, dict]:
    """Create an explicit, practical CPU profile without changing the full study."""

    experiment_cfg = copy.deepcopy(experiment_cfg)
    experiment_cfg["experiment"].update(
        {
            "num_clients": 2,
            "client_sampling_ratio": 1.0,
            "max_rounds": 1,
            "seeds": [experiment_cfg["experiment"]["seeds"][0]],
            "scheduler_treatments": ["tradefl_dynamic"],
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


def scheduler_treatments(exp: dict, selected: list[str] | None = None) -> list[str | None]:
    """Return validated online treatments in deterministic configured order."""

    if "scheduler_treatments" not in exp:
        return [None]
    configured = list(exp["scheduler_treatments"])
    unknown = set(configured) - set(ONLINE_POLICIES)
    if unknown:
        raise ValueError(f"Unknown scheduler treatments: {sorted(unknown)}")
    requested = set(selected or [])
    if requested - set(configured):
        raise ValueError(f"Requested scheduler policies are not configured: {sorted(requested - set(configured))}")
    return [policy for policy in configured if not requested or policy in requested]


def run_model_experiment(
    model_config,
    training,
    exp,
    dataset,
    seed,
    round_path: Path,
    on_round: Callable[[dict], None] | None = None,
    scheduling_policy: str | None = None,
    constraints: dict | None = None,
) -> dict:
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
    shadow_config = exp.get("shadow_pricing", {})
    client_utilities = {client: float(len(records)) for client, records in enumerate(client_records)}
    shadow_scheduler = None
    if scheduling_policy is not None:
        shadow_scheduler = build_online_scheduler(
            scheduling_policy, num_clients, clients_per_round, shadow_config, seed, client_utilities,
        )
    elif shadow_config.get("enabled", False):
        shadow_scheduler = ShadowPriceScheduler(num_clients, clients_per_round, shadow_config, seed)
        scheduling_policy = "tradefl_dynamic"


    for round_index in range(int(exp["max_rounds"])):
        energy_meter = EnergyMeter()
        energy_meter.start()
        if shadow_scheduler is None:
            selected_clients = sorted(rng.choice(num_clients, size=clients_per_round, replace=False).tolist())
            shadow_round = None
        else:
            selected_clients, shadow_round = shadow_scheduler.select_clients()
        downloads = tensor_state_nbytes(global_state) * len(selected_clients)
        updates = []
        compute_seconds = 0.0
        peak_memory = 0
        uploaded = 0
        client_demands = {}
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
            client_demands[client_id] = {
                "peak_memory_bytes": update.peak_accelerator_memory_bytes,
                "compute_time_seconds": update.compute_seconds,
                "communication_bytes": update.uploaded_bytes + update.downloaded_bytes + tensor_state_nbytes(global_state),
            }
            print(f"COMPLETED client={client_id + 1}/{num_clients} in {update.compute_seconds:.1f}s", flush=True)
        global_state = sample_weighted_fedavg(updates)
        validation = trainer.evaluate(dataset.validation, global_state)
        test = trainer.evaluate(dataset.test, global_state)
        latency = time.perf_counter() - round_started
        energy_joules = energy_meter.stop()
        if shadow_scheduler is not None:
            reconciliation = shadow_scheduler.reconcile(client_demands)
            shadow_round.update(reconciliation)
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
            "energy_joules": energy_joules,
            "training_mode": "real_federated",
            "aggregation": "FedAvg",

            "scheduler_policy": scheduling_policy or "seeded_random",
            "selected_action_id": (
                shadow_round.get("selected_action_id", "clients:" + ",".join(map(str, selected_clients)))
                if shadow_round is not None else "clients:" + ",".join(map(str, selected_clients))
            ),
            "target_quality": float(exp["target_quality"]),
            "constraint_provenance": json.dumps(constraints or {}, sort_keys=True),
            "shadow_pricing": shadow_round,

        }
        with round_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        action_record = {
            "plan_id": row["plan_id"], "base_experiment_id": model_config.get("base_experiment_id"),
            "seed": seed, "round_index": round_index, "scheduler_policy": row["scheduler_policy"],
            "selected_action_id": row["selected_action_id"], "selected_clients": selected_clients,
            "target_quality": row["target_quality"], "constraint_provenance": row["constraint_provenance"],
        }
        with (round_path.parent / "online_scheduler_actions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(action_record) + "\n")
        if shadow_round is not None:
            price_record = {
                **action_record,
                "prices_before": shadow_round.get("prices_before", {}),
                "prices_after": shadow_round.get("prices_after", {}),
                "predicted_demand": shadow_round.get("predicted_demand", {}),
                "realized_demand": shadow_round.get("realized_demand", {}),
                "availability": shadow_round.get("availability", {}),
            }
            with (round_path.parent / "online_shadow_price_trajectory.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(price_record) + "\n")

        rounds.append(row)
        if row["validation_utility"] >= float(exp["target_quality"]):
            target_reached = True
            rounds_to_target = round_index + 1
            break
        partial_summary = summarize_model_run(model_config, exp, dataset.name, seed, rounds, target_reached, rounds_to_target)
        if on_round is not None:
            on_round(partial_summary)
        if target_reached:
            break

    return summarize_model_run(model_config, exp, dataset.name, seed, rounds, target_reached, rounds_to_target)


def summarize_model_run(model_config, exp, dataset_name, seed, rounds, target_reached, rounds_to_target) -> dict:
    """Build a CSV-ready summary for a completed or in-progress model run."""

    final = rounds[-1]
    return {
        "plan_id": model_config["experiment_id"],
        "model_id": model_config["model_id"],
        "method": model_config.get("method", "full_finetuning"),
        "seed": seed,
        "dataset": dataset_name,
        "training_mode": "real_federated",
        "aggregation": "FedAvg",
        "scheduler_policy": final.get("scheduler_policy", "seeded_random"),
        "base_experiment_id": model_config.get("base_experiment_id", model_config["experiment_id"]),
        "target_quality": float(exp["target_quality"]),
        "constraint_provenance": final.get("constraint_provenance", "{}"),
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
        "attained_time_to_target_seconds": (
            sum(row["latency_seconds"] for row in rounds) if target_reached else None
        ),
        "censored_observation_horizon_seconds": (
            None if target_reached else sum(row["latency_seconds"] for row in rounds)
        ),
        "mean_round_latency_seconds": sum(row["latency_seconds"] for row in rounds) / len(rounds),
        "privacy_risk": final["privacy_risk"],
        "shadow_pricing_enabled": final.get("shadow_pricing") is not None,
        "shadow_prices_final": (
            json.dumps(final["shadow_pricing"]["prices_after"], sort_keys=True)
            if final.get("shadow_pricing") is not None
            else None
        ),
        "energy_to_target_joules": (
            sum(float(row["energy_joules"]) for row in rounds)
            if all(row["energy_joules"] is not None for row in rounds)
            else None
        ),
    }


def write_live_reports(rows, output: Path, args, experiment_cfg, models_cfg) -> None:
    """Atomically refresh raw metrics, selection artifacts, summaries, and graphs."""

    raw_path = output / "raw_metrics.csv"
    temporary_raw = output / ".raw_metrics.csv.tmp"
    pd.DataFrame(rows).to_csv(temporary_raw, index=False)
    temporary_raw.replace(raw_path)

    budget_cfg = load_yaml(args.budgets)
    weight_cfg = load_yaml(args.weights)
    if "budgets" not in budget_cfg or "scenarios" not in weight_cfg or "constraints" not in experiment_cfg:
        pd.DataFrame(rows).to_csv(output / "plan_summary.csv", index=False)
        return
    scenario = weight_cfg.get("default_scenario", "equal")
    weights = TradeFLWeights(**weight_cfg["scenarios"][scenario])
    if getattr(args, "weight_method", "configured") == "entropy":
        weights = entropy_weights(rows, budget_cfg.get("enabled_metrics", {}), weights)
    result = select_best(
        rows,
        Budgets(**budget_cfg["budgets"]),
        weights,
        Constraints(**experiment_cfg["constraints"]),
        budget_cfg.get("enabled_metrics", {}),
        experiment_cfg["experiment"].get("reference_utility", 1.0),
    )
    write_selection(result, output)
    # Model-plan graphs retain the original model-only x-axis. Scheduler
    # treatments are compared in their dedicated scheduler figures.
    expected_ids = [item["experiment_id"] for item in models_cfg["federated_experiments"]]
    expected_status = {plan_id: "not_run" for plan_id in expected_ids}
    for item in models_cfg.get("external_models", []):
        plan_id = str(item.get("experiment_id", item["model_id"]))
        expected_ids.append(plan_id)
        expected_status[plan_id] = "external_not_federated"
    summarize_results(
        output / "plan_summary.csv",
        output / "summary",
        expected_plan_ids=expected_ids,
        expected_status=expected_status,
        constraints=experiment_cfg.get("constraints", {}),
    )



if __name__ == "__main__":
    main()
