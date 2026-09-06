#!/usr/bin/env python
"""Run real federated training, plan selection, evaluation summary, and graphs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_sibling_select_plan():
    """Load this checkout's sibling script, avoiding unrelated `scripts` packages."""

    path = SCRIPT_DIR / "select_plan.py"
    spec = importlib.util.spec_from_file_location("tradefl_pipeline_select_plan", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load pipeline component: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selection_api_version() -> int:
    """Return the sibling selection API, treating pre-versioned scripts as v1."""

    return int(getattr(_load_sibling_select_plan(), "SELECT_PLAN_API_VERSION", 1))


def validate_component_versions() -> None:
    """Fail before training when selection cannot honor the pipeline contract."""

    version = selection_api_version()
    if version < 3:
        raise SystemExit(
            "Pipeline selection requires --output-dir and --weight-method support. "
            "Update the whole repository so scripts/select_plan.py exposes API v3."
        )



def materialize_selection_config(experiment_config: str, output_dir: str) -> Path:
    """Write a compatibility config whose selection output matches the pipeline."""

    from tradefl.utils.config import load_yaml

    config = load_yaml(REPO_ROOT / experiment_config if not Path(experiment_config).is_absolute() else experiment_config)
    config["experiment"]["output_dir"] = output_dir
    output = REPO_ROOT / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "pipeline_selection_config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def pipeline_commands(args: argparse.Namespace, select_api: int = 3) -> list[list[str]]:
    """Build the three auditable pipeline commands in execution order."""

    python = sys.executable
    output = Path(args.output_dir)
    commands: list[list[str]] = []
    if not args.skip_training:
        training = [
            python,
            str(SCRIPT_DIR / "run_real_federated_experiment.py"),
            "--experiment-config",
            args.experiment_config,
            "--models-config",
            args.models_config,
            "--output-dir",
            str(output),
            "--budgets",
            args.budgets,
            "--weights",
            args.weights,
            "--weight-method",
            args.weight_method,
        ]
        for experiment_id in args.experiment_id or []:
            training.extend(["--experiment-id", experiment_id])
        for policy in getattr(args, "scheduler_policy", None) or []:
            training.extend(["--scheduler-policy", policy])
        for population in getattr(args, "full_participation_clients", None) or []:
            training.extend(["--full-participation-clients", str(population)])
        if getattr(args, "full_participation_clients", None):
            training.extend(["--dirichlet-alpha", str(getattr(args, "dirichlet_alpha", 0.5))])
        if args.cpu_smoke_test:
            training.append("--cpu-smoke-test")
        if args.allow_slow_cpu:
            training.append("--allow-slow-cpu")
        if args.strict_hardware:
            training.append("--strict-hardware")
        if getattr(args, "run_external_baselines", False):
            training.append("--run-external-baselines")
        if getattr(args, "external_only", False):
            training.append("--external-only")
        for external_id in getattr(args, "external_baseline_id", None) or []:
            training.extend(["--external-baseline-id", external_id])
        if getattr(args, "append_results", False):
            training.append("--append-results")
        commands.append(training)
    selection = [
                python,
                str(SCRIPT_DIR / "select_plan.py"),
                "--results",
                str(output / "raw_metrics.csv"),
                "--budgets",
                args.budgets,
                "--weights",
                args.weights,
                "--config",
                str(getattr(args, "selection_config", args.experiment_config)),
            ]
    if select_api >= 2:
        selection.extend(["--output-dir", str(output)])
    if select_api >= 3:
        selection.extend(["--weight-method", args.weight_method])
    elif args.weight_method != "configured":
        raise SystemExit(
            "Entropy weighting requires select_plan API v3. Update scripts/select_plan.py, "
            "or rerun with --weight-method configured."
        )
    commands.extend(
        [
            selection,
            [
                python,
                str(SCRIPT_DIR / "summarize_results.py"),
                "--input",
                str(output / "plan_summary.csv"),
                "--output",
                str(output / "summary"),
                "--models-config",
                args.models_config,
                "--config",
                str(getattr(args, "selection_config", args.experiment_config)),
            ],
        ]
    )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", default="configs/experiment_pubmedqa_real.yaml")
    parser.add_argument("--models-config", default="configs/real_federated_models.yaml")
    parser.add_argument("--output-dir", default="outputs/real_federated")
    parser.add_argument("--budgets", default="configs/budgets.yaml")
    parser.add_argument("--weights", default="configs/weights.yaml")
    parser.add_argument("--weight-method", choices=["configured", "entropy"], default="configured")
    parser.add_argument("--experiment-id", action="append")
    parser.add_argument(
        "--scheduler-policy", action="append",
        choices=["tradefl_dynamic", "tradefl_fixed", "independent", "static_weighted_sum", "greedy"],
<<<<<<< HEAD
    )
    parser.add_argument(
        "--full-participation-clients", action="append", type=int,
        help="Run a no-selection full-participation baseline; repeat for populations such as 10 and 50.",
    )
    parser.add_argument(
        "--dirichlet-alpha", type=float, default=0.5,
        help="Label-skew concentration for the full-participation client populations.",
    )
=======
        )
>>>>>>> 73b8b61 (most probems fixed except dynamic utility comparison)
    parser.add_argument("--cpu-smoke-test", action="store_true")
    parser.add_argument("--allow-slow-cpu", action="store_true")
    parser.add_argument("--strict-hardware", action="store_true")
    parser.add_argument(
        "--run-external-baselines",
        action="store_true",
        help="Run configured paid centralized API baselines after federated training.",
    )
    parser.add_argument(
        "--external-only",
        action="store_true",
        help="Skip federated training and run only configured centralized API baselines.",
    )

    parser.add_argument("--external-baseline-id", action="append")
    parser.add_argument(
        "--append-results",
        action="store_true",
        help="Add an external-only baseline to existing metrics and preserve federated round logs.",
    )
    parser.add_argument("--skip-training", action="store_true", help="Select and graph an existing raw_metrics.csv.")
    args = parser.parse_args()

    validate_component_versions()
    select_api = selection_api_version()
    args.selection_config = str(materialize_selection_config(args.experiment_config, args.output_dir))
    if select_api < 3:
        print(
            f"Compatibility mode: select_plan API v{select_api}; using configured weights and a generated output config.",
            file=sys.stderr,
        )
    commands = pipeline_commands(args, select_api=select_api)
    execute_commands(commands)


def execute_commands(commands: list[list[str]]) -> None:
    """Execute pipeline stages without leaking a subprocess Python traceback."""

    for index, command in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        completed = subprocess.run(command, check=False, cwd=REPO_ROOT)
        if completed.returncode:
            if index == 1 and "--strict-hardware" in command:
                raise SystemExit(
                    "Federated training stopped during hardware validation. "
                    "The selected catalog contains CUDA-only models, but CUDA is unavailable. "
                    "Run the CPU pipeline check with --cpu-smoke-test, or rerun this strict command on a CUDA host."
                )
            raise SystemExit(f"Pipeline stage {index}/{len(commands)} failed with exit code {completed.returncode}.")


if __name__ == "__main__":
    main()
