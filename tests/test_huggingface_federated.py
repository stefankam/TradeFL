import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_real_federated_experiment as runner
from tradefl.data.loaders import DatasetRecord
from tradefl.federation.huggingface import (
    ClientUpdate,
    HuggingFaceClientTrainer,
    UnsupportedRuntimeError,
    concise_native_failure,
    cuda_backend_reason,
    pubmedqa_prompt,
    resolve_runtime,
    tensor_state_nbytes,
    translate_backend_exception,
    unavailable_runtime_reason,
)


def test_pubmedqa_prompt_constrains_output_labels():
    record = DatasetRecord("question: Does it work?\ncontext: Trial text", "yes", {})

    prompt = pubmedqa_prompt(record)

    assert "Does it work?" in prompt
    assert "yes, no, or maybe" in prompt
    assert prompt.endswith("answer:")


def test_tensor_state_bytes_are_real_array_payload_bytes():
    state = {"a": np.zeros((2, 3), dtype=np.float32), "b": np.zeros(4, dtype=np.int16)}

    assert tensor_state_nbytes(state) == 32


def test_runtime_automatically_uses_cpu_and_disables_gpu_precision():
    class Cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def is_bf16_supported():
            return False

    runtime = resolve_runtime(SimpleNamespace(cuda=Cuda()), {"use_cpu": None, "bf16": True, "fp16": True}, "full_finetuning")

    assert runtime == {"use_cpu": True, "bf16": False, "fp16": False}


def test_qlora_fails_early_without_cuda():
    cuda = SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False)

    with pytest.raises(UnsupportedRuntimeError, match="QLoRA 4-bit training requires CUDA"):
        resolve_runtime(SimpleNamespace(cuda=cuda), {"use_cpu": None}, "qlora")


def test_large_model_can_declare_cuda_requirement():
    cuda = SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False)

    with pytest.raises(UnsupportedRuntimeError, match="requires_cuda=true"):
        resolve_runtime(SimpleNamespace(cuda=cuda), {"use_cpu": None}, "lora", requires_cuda=True)


def test_preflight_explains_cpu_compatible_command(monkeypatch):
    torch_module = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr("importlib.import_module", lambda name: torch_module)

    reason = unavailable_runtime_reason({"method": "qlora", "requires_cuda": True}, {"use_cpu": None})

    assert reason == "model requires CUDA; on CPU run --experiment-id bio_clinicalbert_baseline"


def test_cuda_preflight_rejects_detected_but_unusable_runtime():
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, synchronize=lambda: None),
        empty=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("driver mismatch")),
    )

    reason = cuda_backend_reason(torch_module)

    assert reason == "PyTorch detected CUDA but a CUDA tensor operation failed: RuntimeError: driver mismatch"


def test_qlora_preflight_uses_basic_cuda_execution(monkeypatch):
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, synchronize=lambda: None),
        empty=lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr("importlib.import_module", lambda name: torch_module)

    assert unavailable_runtime_reason({"method": "qlora"}, {"use_cpu": None}) is None


def test_lora_preflight_also_uses_basic_cuda_execution(monkeypatch):
    torch_module = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    calls = []
    monkeypatch.setattr("importlib.import_module", lambda name: torch_module)
    monkeypatch.setattr(
        "tradefl.federation.huggingface.cuda_backend_reason",
        lambda torch: calls.append(torch) or None,
    )

    assert unavailable_runtime_reason({"method": "lora"}, {"use_cpu": None}) is None
    assert calls == [torch_module]


def test_native_linker_traceback_is_reduced_to_one_actionable_line():
    traceback_text = """Traceback (most recent call last):
  File \"driver.py\", line 66, in __init__
subprocess.CalledProcessError: gcc -l:libcuda.so.1 returned non-zero exit status 1
"""

    assert concise_native_failure(traceback_text) == "gcc could not link the NVIDIA driver library libcuda.so.1"


def test_missing_python_header_is_reported_before_linker_command_noise():
    output = "fatal error: Python.h: No such file or directory\ngcc command contains -l:libcuda.so.1"

    assert concise_native_failure(output) == "the native compiler cannot find Python.h"


def test_missing_python_header_translation_has_install_remedy():
    error = RuntimeError("Triton fatal error: Python.h: No such file or directory")

    translated = translate_backend_exception(error, "client-local optimizer execution")

    assert "cannot find Python.h" in str(translated)
    assert "python3.12-dev" in str(translated)


def test_late_triton_exception_is_translated_without_native_traceback():
    try:
        raise RuntimeError("triton/runtime build cannot link libcuda.so.1")
    except RuntimeError as exc:
        translated = translate_backend_exception(exc, "client-local optimizer execution")

    assert isinstance(translated, UnsupportedRuntimeError)
    assert "stopped before producing metrics" in str(translated)
    assert "libcuda.so.1" in str(translated)


def test_unrelated_training_exception_is_not_hidden():
    error = ValueError("invalid labels")

    assert translate_backend_exception(error, "training") is None


def test_trainer_rebuilds_missing_runtime_for_backward_compatibility():
    cuda = SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False)
    trainer = object.__new__(HuggingFaceClientTrainer)
    trainer.torch = SimpleNamespace(cuda=cuda)
    trainer.training = {"use_cpu": None, "bf16": True, "fp16": True}
    trainer.model_config = {"method": "full_finetuning"}

    runtime = trainer._resolved_runtime()

    assert runtime == {"use_cpu": True, "bf16": False, "fp16": False}
    assert trainer.runtime == runtime


def test_end_to_end_runner_aggregates_clients_and_writes_real_provenance(tmp_path, monkeypatch):
    class FakeTrainer:
        def __init__(self, model_config, labels, training):
            self.labels = labels

        def initial_state(self):
            return {"adapter": np.zeros(1, dtype=np.float32)}

        def train_client(self, records, global_state):
            value = float(sum(record.metadata["index"] for record in records))
            state = {"adapter": np.array([value], dtype=np.float32)}
            return ClientUpdate(state, len(records), 0.1, 128, tensor_state_nbytes(state))

        def evaluate(self, records, global_state):
            return {"accuracy": 0.75, "macro_f1": 0.7}

    monkeypatch.setattr(runner, "HuggingFaceClientTrainer", FakeTrainer)
    records = [DatasetRecord(f"example {index}", "yes" if index % 2 else "no", {"index": index}) for index in range(10)]
    dataset = SimpleNamespace(train=records, validation=records[:2], test=records[2:4], labels=("no", "yes"), name="pubmedqa")
    round_path = tmp_path / "round_metrics.jsonl"

    summary = runner.run_model_experiment(
        {"experiment_id": "test_model", "model_id": "test/model", "architecture": "causal_lm", "method": "lora"},
        {},
        {
            "num_clients": 2,
            "client_sampling_ratio": 1.0,
            "max_rounds": 1,
            "target_quality": 0.7,
            "primary_metric": "accuracy",
            "reference_utility": 0.9,
        },
        dataset,
        42,
        round_path,
    )

    row = json.loads(round_path.read_text())
    assert row["training_mode"] == "real_federated"
    assert row["selected_clients"] == [0, 1]
    assert row["bytes_uploaded"] == 8
    assert row["bytes_downloaded"] == 8
    assert summary["training_mode"] == "real_federated"
    assert summary["target_reached"] is True


def test_cpu_smoke_profile_is_explicitly_bounded():
    experiment = {"experiment": {"num_clients": 5, "max_rounds": 5, "seeds": [42, 43, 44]}}
    training = {"batch_size": 1, "gradient_accumulation_steps": 8, "max_length": 512, "use_cpu": None}

    smoke_experiment, smoke_training = runner.apply_cpu_smoke_profile(experiment, training)

    assert smoke_experiment["experiment"]["num_clients"] == 2
    assert smoke_experiment["experiment"]["max_rounds"] == 1
    assert smoke_experiment["experiment"]["seeds"] == [42]
    assert smoke_training == {
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_length": 128,
        "evaluation_batch_size": 8,
        "max_train_samples_per_client": 8,
        "cpu_threads": 4,
        "use_cpu": True,
        "local_epochs": 1,
    }
    assert experiment["experiment"]["num_clients"] == 5


def test_full_transformer_training_requires_explicit_cpu_opt_in(monkeypatch):
    torch_module = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr("importlib.import_module", lambda name: torch_module)

    reason = unavailable_runtime_reason({"method": "full_finetuning"}, {"use_cpu": None})

    assert "--cpu-smoke-test" in reason
    assert unavailable_runtime_reason(
        {"method": "full_finetuning"}, {"use_cpu": None, "allow_slow_cpu": True}
    ) is None


def test_main_skips_cuda_model_and_continues_cpu_baseline(tmp_path, monkeypatch):
    class FakeTrainer:
        def __init__(self, model_config, labels, training):
            if model_config.get("requires_cuda"):
                raise UnsupportedRuntimeError("CUDA required")

        def initial_state(self):
            return {"weight": np.zeros(1, dtype=np.float32)}

        def train_client(self, records, global_state):
            return ClientUpdate({"weight": np.ones(1, dtype=np.float32)}, len(records), 0.1, 0, 4)

        def evaluate(self, records, global_state):
            return {"accuracy": 1.0, "macro_f1": 1.0}

    records = [DatasetRecord(f"example {index}", "yes" if index % 2 else "no", {"index": index}) for index in range(4)]
    dataset = SimpleNamespace(train=records, validation=records[:2], test=records[2:], labels=("no", "yes"), name="pubmedqa")
    experiment = {
        "experiment": {
            "num_clients": 2,
            "client_sampling_ratio": 1.0,
            "local_epochs": 1,
            "max_rounds": 1,
            "target_quality": 0.5,
            "primary_metric": "accuracy",
            "reference_utility": 1.0,
            "seeds": [42],
        },
        "dataset": {"name": "pubmedqa"},
    }
    models = {
        "training": {},
        "federated_experiments": [
            {"experiment_id": "gpu", "model_id": "gpu/model", "requires_cuda": True},
            {"experiment_id": "cpu", "model_id": "cpu/model"},
        ],
    }
    monkeypatch.setattr(runner, "HuggingFaceClientTrainer", FakeTrainer)
    monkeypatch.setattr(runner, "ensure_dataset_available", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "load_dataset_bundle", lambda config: dataset)
    monkeypatch.setattr(runner, "load_yaml", lambda path: models if "models" in str(path) else experiment)
    monkeypatch.setattr(
        runner,
        "unavailable_runtime_reason",
        lambda model_config, training: "CUDA required" if model_config.get("requires_cuda") else None,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["run_real_federated_experiment.py", "--output-dir", str(tmp_path)],
    )

    runner.main()

    skipped = json.loads((tmp_path / "skipped_experiments.json").read_text())
    assert skipped == [{"experiment_id": "gpu", "seed": None, "reason": "CUDA required"}]
    summaries = pd.read_csv(tmp_path / "raw_metrics.csv")
    assert summaries["plan_id"].tolist() == ["cpu"]
