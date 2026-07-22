"""Simulation loop for TokenOS."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import json
from pathlib import Path
from .device import EdgeDevice
from .tokens import ResourceTokenManager
from .optimizer import TokenOptimizer
from .baselines import BaselineRunner
from .compat import rng as make_rng, dataframe

@dataclass
class Simulation:
    rounds: int = 100
    num_devices: int = 50
    seed: int = 42

    def create_devices(self) -> List[EdgeDevice]:
        rng = make_rng(self.seed)
        devices=[]
        for i in range(self.num_devices):
            devices.append(EdgeDevice(f"device_{i:03d}", float(rng.uniform(2,16)), float(rng.uniform(1,12)), float(rng.uniform(2,100)),
                                      float(rng.uniform(25,100)), float(rng.uniform(20,200)), float(rng.uniform(1,10)),
                                      float(rng.uniform(8,256)), float(rng.uniform(1,25))))
        ResourceTokenManager().initialize(devices)
        return devices

    def run(self, output: str) -> Dict[str, object]:
        out = Path(output); out.mkdir(parents=True, exist_ok=True)
        devices = self.create_devices(); rng=make_rng(self.seed)
        systems = {"TokenOS": TokenOptimizer(), "Random": BaselineRunner("Random","random",self.seed), "FixedSplit": BaselineRunner("FixedSplit","fixed_split",self.seed),
                   "NoReallocation": BaselineRunner("NoReallocation","no_reallocation",self.seed), "LocalOnly": BaselineRunner("LocalOnly","local_only",self.seed),
                   "CloudOnly": BaselineRunner("CloudOnly","cloud_only",self.seed)}
        rows=[]; plans={name:[] for name in systems}
        for r in range(self.rounds):
            for d in devices: d.randomize_runtime(rng)
            for name, system in systems.items():
                plan = system.plan_round(devices, r)
                plans[name].append(plan)
                row={"round":r,"system":name}; row.update(plan["metrics"]); rows.append(row)
        df=dataframe(rows); df.to_csv(out/"metrics.csv", index=False)
        (out/"execution_plans.json").write_text(json.dumps(plans, indent=2), encoding="utf-8")
        return {"metrics": df}
