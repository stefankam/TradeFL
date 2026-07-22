"""Optional dependency helpers for constrained environments."""
from __future__ import annotations
import csv, importlib.util, json, random
from pathlib import Path
from typing import Any, Dict, Iterable, List

class RandomGenerator:
    def __init__(self, seed: int): self._r = random.Random(seed)
    def uniform(self, a: float, b: float) -> float: return self._r.uniform(a, b)
    def normal(self, mean: float, sd: float) -> float: return self._r.gauss(mean, sd)
    def choice(self, seq, size: int, replace: bool = False): return self._r.sample(list(seq), size) if not replace else [self._r.choice(seq) for _ in range(size)]

def rng(seed: int) -> Any:
    if importlib.util.find_spec("numpy"):
        import numpy as np
        return np.random.default_rng(seed)
    return RandomGenerator(seed)

class SimpleDataFrame:
    def __init__(self, rows: List[Dict[str, Any]]): self.rows = rows
    def to_csv(self, path: str | Path, index: bool = False) -> None:
        keys = list(self.rows[0].keys()) if self.rows else []
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(self.rows)
    def groupby(self, key: str):
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in self.rows: groups.setdefault(row[key], []).append(row)
        for name, rows in groups.items(): yield name, SimpleDataFrame(rows)
    def __getitem__(self, key: str): return [row[key] for row in self.rows]

def dataframe(rows: List[Dict[str, Any]]) -> Any:
    if importlib.util.find_spec("pandas"):
        import pandas as pd
        return pd.DataFrame(rows)
    return SimpleDataFrame(rows)
