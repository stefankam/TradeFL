"""Incremental metrics writers for auditable experiments."""
from __future__ import annotations
import csv, json
from pathlib import Path
class MetricsLogger:
    def __init__(self, output_dir:str|Path='outputs') -> None:
        self.output_dir=Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
    def append_jsonl(self, filename:str, row:dict)->None:
        with (self.output_dir/filename).open('a',encoding='utf-8') as f: f.write(json.dumps(row, default=str)+'\n')
    def write_csv(self, filename:str, rows:list[dict])->None:
        if not rows: return
        with (self.output_dir/filename).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction='ignore'); w.writeheader(); w.writerows(rows)
