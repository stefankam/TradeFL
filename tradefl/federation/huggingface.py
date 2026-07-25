"""Real Hugging Face/PEFT client training for sequential federated simulation."""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from tradefl.data.loaders import DatasetRecord


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

    @property
    def uses_adapter(self) -> bool:
        return self.model_config.get("method") in {"lora", "qlora"}

    def initial_state(self) -> dict[str, np.ndarray]:
        model, _ = self._load_model_and_tokenizer()
        state = self._extract_trainable_state(model)
        self._release(model)
        return state

    def train_client(self, records: list[DatasetRecord], global_state: dict[str, np.ndarray]) -> ClientUpdate:
        model, tokenizer = self._load_model_and_tokenizer()
        self._load_trainable_state(model, global_state)
        dataset = self._tokenize(records, tokenizer)
        if self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        trainer = self.transformers.Trainer(
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
                fp16=bool(self.training.get("fp16", False)),
                bf16=bool(self.training.get("bf16", False)),
                remove_unused_columns=False,
            ),
            train_dataset=dataset,
            data_collator=self._collator(tokenizer),
        )
        trainer.train()
        elapsed = time.perf_counter() - started
        peak = int(self.torch.cuda.max_memory_allocated()) if self.torch.cuda.is_available() else 0
        state = self._extract_trainable_state(model)
        uploaded = tensor_state_nbytes(state)
        self._release(model, trainer)
        return ClientUpdate(state, len(records), elapsed, peak, uploaded)

    def evaluate(self, records: list[DatasetRecord], global_state: dict[str, np.ndarray]) -> dict[str, float]:
        model, tokenizer = self._load_model_and_tokenizer()
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
        model_id = self.model_config["model_id"]
        revision = self.model_config.get("revision")
        common = {"revision": revision} if revision else {}
        tokenizer = self.transformers.AutoTokenizer.from_pretrained(model_id, **common)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization_config = None
        if self.model_config.get("method") == "qlora":
            quantization_config = self.transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(self.torch, self.training.get("compute_dtype", "bfloat16")),
            )
        load_kwargs = {**common, "device_map": self.training.get("device_map", "auto")}
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
            task_type = "SEQ_CLS" if self.model_config["architecture"] == "sequence_classification" else "CAUSAL_LM"
            lora = self.peft.LoraConfig(
                task_type=task_type,
                r=int(self.model_config.get("lora_rank", 8)),
                lora_alpha=int(self.model_config.get("lora_alpha", 16)),
                lora_dropout=float(self.model_config.get("lora_dropout", 0.05)),
                target_modules=self.model_config.get("target_modules"),
            )
            model = self.peft.get_peft_model(model, lora)
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
        state = self.peft.get_peft_model_state_dict(model) if self.uses_adapter else {
            name: tensor for name, tensor in model.state_dict().items() if tensor.is_floating_point()
        }
        return {name: tensor.detach().cpu().numpy().copy() for name, tensor in state.items()}

    def _load_trainable_state(self, model, state: dict[str, np.ndarray]) -> None:
        tensors = {name: self.torch.from_numpy(value) for name, value in state.items()}
        if self.uses_adapter:
            self.peft.set_peft_model_state_dict(model, tensors)
        else:
            model.load_state_dict(tensors, strict=False)

    def _evaluate_classifier(self, model, tokenizer, records):
        predictions = []
        device = next(model.parameters()).device
        with self.torch.no_grad():
            for record in records:
                batch = tokenizer(pubmedqa_prompt(record), return_tensors="pt", truncation=True, max_length=int(self.training.get("max_length", 512)))
                batch = {key: value.to(device) for key, value in batch.items()}
                prediction = int(model(**batch).logits.argmax(dim=-1).item())
                predictions.append(self.labels[prediction])
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
