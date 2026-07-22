"""Baseline schedulers for comparison."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from .device import EdgeDevice
from .optimizer import TokenOptimizer
from .tokens import ResourceTokenManager
from .compat import rng as make_rng

@dataclass
class BaselineRunner:
    name: str
    mode: str
    seed: int = 0

    def plan_round(self, devices: List[EdgeDevice], round_id: int) -> Dict:
        rng = make_rng(self.seed + round_id)
        tm = ResourceTokenManager()
        if self.mode != "no_reallocation":
            for d in devices: tm.update_allocation(d)
        selected = list(rng.choice(devices, size=max(1, len(devices)//5), replace=False)) if self.mode == "random" else devices[:max(1, len(devices)//5)]
        opt = TokenOptimizer(tm)
        device_plans=[]; totals={"utility":0,"accuracy":0,"energy":0,"latency":0,"memory_pressure":0,"communication":0}
        for d in selected:
            t=d.current_token_budget or tm.update_allocation(d)
            exec_mode = "local SLM only" if self.mode == "local_only" else "cloud LLM only" if self.mode == "cloud_only" else "hybrid SLM + cloud LLM"
            m=opt.score(t, exec_mode, "keep activation", {"dvfs_level":"medium"})
            for k in totals: totals[k]+=m[k]
            device_plans.append({"device_id":d.device_id,"tokens":t,"split_layer":6,"memory_strategy":"keep activation","inference_config":{"model":"quantized model","lora_rank":"medium","dvfs_level":"medium"},"execution_mode":exec_mode,"metrics":m})
        n=max(1,len(selected)); metrics={k:v/n for k,v in totals.items()}; metrics["successful_clients"]=len(selected)
        return {"round":round_id,"selected_clients":[d.device_id for d in selected],"device_plans":device_plans,"metrics":metrics}
