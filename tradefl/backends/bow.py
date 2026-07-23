"""A real, dependency-light text-classification fine-tuning backend.

This backend trains on actual dataset records rather than synthetic metrics. It is
intended as a runnable reference path for CI and small local experiments. The
public plan interface is backend-agnostic so a Transformers/PEFT backend can be
added without changing selection code.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from tradefl.data.loaders import DatasetRecord

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class BagOfWordsFineTuningBackend:
    """Incrementally fine-tune a multinomial Naive Bayes text classifier."""

    labels: tuple[str, ...]
    class_counts: Counter[str] = field(default_factory=Counter)
    token_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    vocabulary: set[str] = field(default_factory=set)

    def train(self, records: list[DatasetRecord]) -> None:
        """Update model counts from one local training batch."""

        for record in records:
            self.class_counts[record.label] += 1
            tokens = _tokenize(record.text)
            self.token_counts[record.label].update(tokens)
            self.vocabulary.update(tokens)

    def evaluate(self, records: list[DatasetRecord]) -> dict[str, float]:
        """Evaluate accuracy and macro-F1 on labeled records."""

        if not records:
            return {"accuracy": 0.0, "macro_f1": 0.0}
        predictions = [self.predict(record.text) for record in records]
        accuracy = sum(pred == record.label for pred, record in zip(predictions, records)) / len(records)
        f1_values = []
        for label in self.labels:
            tp = sum(pred == label and record.label == label for pred, record in zip(predictions, records))
            fp = sum(pred == label and record.label != label for pred, record in zip(predictions, records))
            fn = sum(pred != label and record.label == label for pred, record in zip(predictions, records))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return {"accuracy": accuracy, "macro_f1": sum(f1_values) / len(f1_values)}

    def predict(self, text: str) -> str:
        """Predict the most likely label for text."""

        if not self.class_counts:
            return self.labels[0]
        total_examples = sum(self.class_counts.values())
        vocab_size = max(1, len(self.vocabulary))
        scores: dict[str, float] = {}
        for label in self.labels:
            prior = (self.class_counts[label] + 1) / (total_examples + len(self.labels))
            label_token_total = sum(self.token_counts[label].values()) + vocab_size
            score = math.log(prior)
            for token in _tokenize(text):
                score += math.log((self.token_counts[label][token] + 1) / label_token_total)
            scores[label] = score
        return max(scores, key=scores.get)

    def serialized_size_bytes(self) -> int:
        """Return serialized model-update size for communication accounting."""

        return len(pickle.dumps({"class_counts": self.class_counts, "token_counts": dict(self.token_counts)}))


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]
