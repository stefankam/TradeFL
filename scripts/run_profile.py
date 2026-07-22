#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse, csv, importlib, json
from pathlib import Path
from tradefl.federation.simulator import run_to_target
from tradefl.utils.config import load_yaml
from tradefl.utils.seeds import set_seed

def build_plan(plan_id:str, seed:int, plans_cfg:list[dict]):
    cfg=next(p for p in plans_cfg if p['plan_id']==plan_id); mod=importlib.import_module(f"tradefl.plans.{cfg['type']}")
    return mod.build(seed=seed, **cfg)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--plan',required=True); ap.add_argument('--seed',type=int,required=True); args=ap.parse_args()
    cfg=load_yaml(args.config); exp=cfg['experiment']; plans=load_yaml('configs/plans.yaml')['plans']; set_seed(args.seed)
    plan=build_plan(args.plan,args.seed,plans); rounds, summary=run_to_target(plan, exp['target_quality'], exp['max_rounds'])
    out=Path(exp.get('output_dir','outputs')); out.mkdir(exist_ok=True)
    with (out/'round_metrics.jsonl').open('a',encoding='utf-8') as f:
        for r in rounds: f.write(json.dumps(r, default=str)+'\n')
    summary.update({'plan_id':args.plan,'seed':args.seed,'accuracy_loss':max(0, exp.get('reference_utility',1.0)-summary['validation_utility'])})
    path=out/'raw_metrics.csv'; exists=path.exists()
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not exists: w.writeheader()
        w.writerow(summary)
if __name__=='__main__': main()
