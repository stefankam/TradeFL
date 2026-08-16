"""Real Hugging Face/PEFT client training for sequential federated simulation."""
from __future__ import annotations

import copy
import importlib
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np

from tradefl.data.loaders import DatasetRecord
from tradefl.plans.distillation import transformer_distillation_trainer
from tradefl.plans.full_finetuning import extract_transformer_state, load_transformer_state
from tradefl.plans.lora import attach_peft_lora
from tradefl.plans.qlora import quantization_config as build_qlora_config
from tradefl.plans.splitfed import register_transformer_boundary


class UnsupportedRuntimeError(RuntimeError):
    """The requested training method cannot run on the detected hardware."""


def pubmedqa_prompt(record: DatasetRecord) -> str:
    """Return the common constrained-answer prompt used by every model family."""

    return f"{record.text}\nAnswer with exactly one label: yes, no, or maybe.\nanswer:"


@dataclass
class ClientUpdate:
    state: dict[str, np.ndarray]
    num_examples: int
    compute_seconds: float
    peak_accelerator_memory_bytes: int
    uploaded_bytes: int
    downloaded_bytes: int = 0


class HuggingFaceClientTrainer:
    """Load, locally train, evaluate, and serialize one real Transformer model."""

    def __init__(self, model_config: dict[str, Any], labels: tuple[str, ...], training: dict[str, Any]) -> None:
        self.model_config = model_config
        self.labels = labels
        self.label_to_id = {label: index for index, label in enumerate(labels)}
        self.training = training
        self.torch = importlib.import_module("torch")
        self.transformers = importlib.import_module("transformers")
        self.datasets = importlib.import_module("datasets")
        self.peft = importlib.import_module("peft")
        self.runtime = resolve_runtime(
            self.torch,
            training,
            str(model_config.get("method", "full_finetuning")),
            requires_cuda=bool(model_config.get("requires_cuda", False)),
        )
        if self.runtime["use_cpu"] and hasattr(self.torch, "set_num_threads"):
            self.torch.set_num_threads(max(1, int(training.get("cpu_threads", 4))))

    def _resolved_runtime(self) -> dict[str, bool]:
        """Return runtime flags, rebuilding them for older/deserialized trainer objects."""

        runtime = getattr(self, "runtime", None)
        if runtime is None:
            runtime = resolve_runtime(
                self.torch,
                self.training,
                str(self.model_config.get("method", "full_finetuning")),
                requires_cuda=bool(self.model_config.get("requires_cuda", False)),
            )
            self.runtime = runtime
        return runtime

    @property
    def uses_adapter(self) -> bool:
        return self.model_config.get("method") in {"lora", "qlora"}

    @property
    def uses_distillation(self) -> bool:
        return self.model_config.get("method") == "federated_distillation"

    @property
    def uses_splitfed(self) -> bool:
        return self.model_config.get("method") == "splitfed"

    def initial_state(self) -> dict[str, np.ndarray]:
        try:
            model, _ = self._load_model_and_tokenizer()
            state = self._extract_trainable_state(model)
            self._release(model)
            return state
        except Exception as exc:
            translated = translate_backend_exception(exc, "initial model loading")
            if translated:
                raise translated from None
            raise

    def train_client(self, records: list[DatasetRecord], global_state: dict[str, np.ndarray]) -> ClientUpdate:
        runtime = self._resolved_runtime()
        model, tokenizer = self._load_model_and_tokenizer()
        self._load_trainable_state(model, global_state)
        teacher = copy.deepcopy(model).eval() if self.uses_distillation else None
        if teacher is not None:
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        transfer_counter = {"uploaded_bytes": 0, "downloaded_bytes": 0}
        split_handles = self._register_split_boundary(model, transfer_counter) if self.uses_splitfed else []
        dataset = self._tokenize(records, tokenizer)
        effective_batch = int(self.training.get("batch_size", 1)) * int(
            self.training.get("gradient_accumulation_steps", 1)
        )
        steps = max(1, int(np.ceil(len(records) / effective_batch)))
        print(
            f"  local optimizer: {steps} steps/epoch, "
            f"sequence_length<={self.training.get('max_length', 512)}, "
            f"device={'CPU' if runtime['use_cpu'] else 'CUDA'}",
            flush=True,
        )
        if self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        trainer_class = self._distillation_trainer_class() if self.uses_distillation else self.transformers.Trainer
        trainer_kwargs = {}
        if teacher is not None:
            trainer_kwargs = {
                "teacher_model": teacher,
                "temperature": float(self.model_config.get("distillation_temperature", 2.0)),
                "alpha": float(self.model_config.get("distillation_alpha", 0.5)),
            }
        trainer = trainer_class(
            model=model,
            args=self.transformers.TrainingArguments(
                output_dir=str(self.training.get("temporary_output_dir", "/tmp/tradefl-client")),
                per_device_train_batch_size=int(self.training.get("batch_size", 1)),
                gradient_accumulation_steps=int(self.training.get("gradient_accumulation_steps", 1)),
                learning_rate=float(self.training.get("learning_rate", 2e-4)),
                num_train_epochs=float(self.training.get("local_epochs", 1)),
                logging_strategy="no",
                save_strategy="no",
                report_to=[],
                use_cpu=runtime["use_cpu"],
                fp16=runtime["fp16"],
                bf16=runtime["bf16"],
                optim="adamw_torch",
                torch_compile=False,
                remove_unused_columns=False,
            ),
            train_dataset=dataset,
            data_collator=self._collator(tokenizer),
            **trainer_kwargs,
        )
        try:
            trainer.train()
        except Exception as exc:
            for handle in split_handles:
                handle.remove()
            self._release(model, trainer, teacher)
            translated = translate_backend_exception(exc, "client-local optimizer execution")
            if translated:
                raise translated from None
            raise
        elapsed = time.perf_counter() - started
        peak = int(self.torch.cuda.max_memory_allocated()) if self.torch.cuda.is_available() else 0
        state = self._extract_trainable_state(model)
        uploaded = tensor_state_nbytes(state) + transfer_counter["uploaded_bytes"]
        for handle in split_handles:
            handle.remove()
        self._release(model, trainer, teacher)
        return ClientUpdate(state, len(records), elapsed, peak, uploaded, transfer_counter["downloaded_bytes"])

    def _distillation_trainer_class(self):
        """Build a Trainer using label CE plus temperature-scaled teacher KL."""

        functional = importlib.import_module("torch.nn.functional")
        return transformer_distillation_trainer(self.transformers.Trainer, self.torch, functional)

    def _register_split_boundary(self, model, counter: dict[str, int]):
        """Measure tensors crossing a real BERT encoder cut in both directions."""

        encoder = getattr(getattr(model, "bert", None), "encoder", None)
        layers = getattr(encoder, "layer", [])
        split_layer = int(self.model_config.get("split_layer", len(layers) // 2))
        return [register_transformer_boundary(model, split_layer, counter)]

    def evaluate(self, records: list[DatasetRecord], global_state: dict[str, np.ndarray]) -> dict[str, float]:
        try:
            model, tokenizer = self._load_model_and_tokenizer()
        except Exception as exc:
            translated = translate_backend_exception(exc, "global-model evaluation loading")
            if translated:
                raise translated from None
            raise
        self._load_trainable_state(model, global_state)
        model.eval()
        predictions = (
            self._evaluate_classifier(model, tokenizer, records)
            if self.model_config["architecture"] == "sequence_classification"
            else self._evaluate_causal_lm(model, tokenizer, records)
        )
        self._release(model)
        correct = sum(prediction == record.label for prediction, record in zip(predictions, records))
        accuracy = correct / len(records)
        f1s = []
        for label in self.labels:
            tp = sum(pred == label and row.label == label for pred, row in zip(predictions, records))
            fp = sum(pred == label and row.label != label for pred, row in zip(predictions, records))
            fn = sum(pred != label and row.label == label for pred, row in zip(predictions, records))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return {"accuracy": accuracy, "macro_f1": sum(f1s) / len(f1s)}

    def _load_model_and_tokenizer(self):
        runtime = self._resolved_runtime()
        model_id = self.model_config["model_id"]
        revision = self.model_config.get("revision")
        common = {"revision": revision} if revision else {}
        tokenizer = self.transformers.AutoTokenizer.from_pretrained(model_id, **common)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization_config = None
        if self.model_config.get("method") == "qlora":
            quantization_config = build_qlora_config(self.transformers, self.torch, self.training)
        load_kwargs = dict(common)
        if not runtime["use_cpu"]:
            load_kwargs["device_map"] = self.training.get("device_map", "auto")
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        if self.model_config["architecture"] == "sequence_classification":
            model = self.transformers.AutoModelForSequenceClassification.from_pretrained(
                model_id,
                num_labels=len(self.labels),
                id2label={index: label for label, index in self.label_to_id.items()},
                label2id=self.label_to_id,
                **load_kwargs,
            )
        else:
            model = self.transformers.AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if self.uses_adapter:
            if quantization_config is not None:
                model = self.peft.prepare_model_for_kbit_training(model)
            model = attach_peft_lora(model, self.peft, self.model_config, self.model_config["architecture"])
        return model, tokenizer

    def _tokenize(self, records: list[DatasetRecord], tokenizer):
        max_length = int(self.training.get("max_length", 512))
        rows = []
        for record in records:
            prompt = pubmedqa_prompt(record)
            if self.model_config["architecture"] == "sequence_classification":
                encoded = tokenizer(prompt, truncation=True, max_length=max_length)
                encoded["labels"] = self.label_to_id[record.label]
            else:
                prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
                answer_ids = tokenizer(" " + record.label + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
                input_ids = (prompt_ids + answer_ids)[:max_length]
                prompt_length = min(len(prompt_ids), len(input_ids))
                encoded = {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": [-100] * prompt_length + input_ids[prompt_length:],
                }
            rows.append(encoded)
        return self.datasets.Dataset.from_list(rows)

    def _collator(self, tokenizer):
        if self.model_config["architecture"] == "sequence_classification":
            return self.transformers.DataCollatorWithPadding(tokenizer)
        return self.transformers.DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    def _extract_trainable_state(self, model) -> dict[str, np.ndarray]:
        if self.uses_adapter:
            state = self.peft.get_peft_model_state_dict(model)
            return {name: tensor.detach().cpu().numpy().copy() for name, tensor in state.items()}
        return extract_transformer_state(model)

    def _load_trainable_state(self, model, state: dict[str, np.ndarray]) -> None:
        if self.uses_adapter:
            tensors = {name: self.torch.from_numpy(value) for name, value in state.items()}
            self.peft.set_peft_model_state_dict(model, tensors)
        else:
            load_transformer_state(model, self.torch, state)

    def _evaluate_classifier(self, model, tokenizer, records):
        predictions = []
        device = next(model.parameters()).device
        batch_size = max(1, int(self.training.get("evaluation_batch_size", 8)))
        with self.torch.no_grad():
            for start in range(0, len(records), batch_size):
                batch_records = records[start : start + batch_size]
                batch = tokenizer(
                    [pubmedqa_prompt(record) for record in batch_records],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(self.training.get("max_length", 512)),
                )
                batch = {key: value.to(device) for key, value in batch.items()}
                predicted_ids = model(**batch).logits.argmax(dim=-1).tolist()
                predictions.extend(self.labels[int(prediction)] for prediction in predicted_ids)
        return predictions

    def _evaluate_causal_lm(self, model, tokenizer, records):
        predictions = []
        device = next(model.parameters()).device
        with self.torch.no_grad():
            for record in records:
                batch = tokenizer(pubmedqa_prompt(record), return_tensors="pt", truncation=True, max_length=int(self.training.get("max_length", 512)))
                batch = {key: value.to(device) for key, value in batch.items()}
                generated = model.generate(**batch, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                answer = tokenizer.decode(generated[0, batch["input_ids"].shape[1] :], skip_special_tokens=True).lower()
                predictions.append(next((label for label in self.labels if label in answer), self.labels[0]))
        return predictions

    def _release(self, *objects) -> None:
        del objects
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def tensor_state_nbytes(state: dict[str, np.ndarray]) -> int:
    """Return actual serialized tensor payload size before transport framing."""

    return sum(array.nbytes for array in state.values())


def translate_backend_exception(exc: Exception, phase: str) -> UnsupportedRuntimeError | None:
    """Translate late native CUDA/Triton failures into skippable runtime errors."""

    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    markers = ("triton/runtime", "Triton", "libcuda.so.1", "CudaUtils", "_create_driver")
    if not any(marker in detail for marker in markers):
        return None
    tail = concise_native_failure(detail)
    remedy = (
        " Install the Python development headers matching this interpreter (for example python3.12-dev)."
        if "Python.h" in detail
        else " Verify libcuda.so.1 and use compatible PyTorch/Triton/bitsandbytes packages."
    )
    return UnsupportedRuntimeError(
        f"{phase} requires an operational Triton/NVIDIA backend, but native initialization failed: {tail}. "
        f"The experiment was stopped before producing metrics.{remedy}"
    )


def cuda_backend_reason(torch_module) -> str | None:
    """Verify that PyTorch can execute a basic CUDA operation."""

    if not bool(torch_module.cuda.is_available()):
        return "PyTorch cannot detect a CUDA GPU"
    try:
        probe = torch_module.empty(1, device="cuda")
        torch_module.cuda.synchronize()
        del probe
    except Exception as exc:
        return f"PyTorch detected CUDA but a CUDA tensor operation failed: {type(exc).__name__}: {exc}"
    return None


def concise_native_failure(output: str) -> str:
    """Reduce a compiler traceback to a stable, actionable one-line reason."""

    if "Python.h: No such file or directory" in output:
        return "the native compiler cannot find Python.h"
    if "libcuda.so.1" in output:
        return "gcc could not link the NVIDIA driver library libcuda.so.1"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    detail = lines[-1] if lines else "unknown native backend failure"
    return detail[-500:]


def unavailable_runtime_reason(model_config: dict[str, Any], training: dict[str, Any]) -> str | None:
    """Return a user-facing hardware incompatibility before loading model weights."""

    torch_module = importlib.import_module("torch")
    cuda_available = bool(torch_module.cuda.is_available())
    use_cpu = not cuda_available if training.get("use_cpu") is None else bool(training.get("use_cpu"))
    if training.get("use_cpu") is False and not cuda_available:
        return "use_cpu=false was requested, but PyTorch cannot detect a CUDA GPU"
    if bool(model_config.get("requires_cuda", False)) and use_cpu:
        return "model requires CUDA; on CPU run --experiment-id bio_clinicalbert_baseline"
    if model_config.get("method") == "qlora" and use_cpu:
        return "QLoRA 4-bit training requires CUDA; on CPU run --experiment-id bio_clinicalbert_baseline"
    if not use_cpu:
        backend_reason = cuda_backend_reason(torch_module)
        if backend_reason:
            return backend_reason
    if use_cpu and not bool(training.get("allow_slow_cpu", False)):
        return (
            "full Transformer training on CPU is disabled because it may show no progress for hours; "
            "run with --cpu-smoke-test, use CUDA, or explicitly accept the cost with --allow-slow-cpu"
        )
    return None


def resolve_runtime(
    torch_module,
    training: dict[str, Any],
    method: str,
    requires_cuda: bool = False,
) -> dict[str, bool]:
    """Resolve CPU/GPU and precision flags without requesting unsupported modes."""

    cuda_available = bool(torch_module.cuda.is_available())
    requested_cpu = training.get("use_cpu")
    use_cpu = not cuda_available if requested_cpu is None else bool(requested_cpu)
    if requested_cpu is False and not cuda_available:
        raise UnsupportedRuntimeError("use_cpu=false was requested, but PyTorch cannot detect a CUDA GPU.")
    if requires_cuda and use_cpu:
        raise UnsupportedRuntimeError("This model is marked requires_cuda=true, but PyTorch cannot detect a CUDA GPU.")
    if method == "qlora" and use_cpu:
        raise UnsupportedRuntimeError(
            "QLoRA 4-bit training requires CUDA; on CPU run --experiment-id bio_clinicalbert_baseline."
        )
    bf16_check = getattr(torch_module.cuda, "is_bf16_supported", lambda: False)
    bf16_supported = cuda_available and bool(bf16_check())
    bf16 = not use_cpu and bool(training.get("bf16", False)) and bf16_supported
    fp16 = not use_cpu and bool(training.get("fp16", False)) and not bf16
    return {"use_cpu": use_cpu, "bf16": bf16, "fp16": fp16}
