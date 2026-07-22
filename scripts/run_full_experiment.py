#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse, importlib, json
from pathlib import Path
import pandas as pd
from tradefl.federation.simulator import run_to_target
from tradefl.utils.config import load_yaml
from tradefl.utils.seeds import set_seed

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args()
    cfg=load_yaml(args.config); exp=cfg['experiment']; plans=load_yaml('configs/plans.yaml')['plans']; out=Path(exp.get('output_dir','outputs')); out.mkdir(exist_ok=True)
    if exp.get('overwrite',False):
        for name in ['raw_metrics.csv','round_metrics.jsonl']: (out/name).unlink(missing_ok=True)
    summaries=[]
    for seed in exp['seeds']:
        set_seed(seed)
        for pcfg in plans:
            mod=importlib.import_module(f"tradefl.plans.{pcfg['type']}"); plan=mod.build(seed=seed, **pcfg)
            rounds, summary=run_to_target(plan, exp['target_quality'], exp['max_rounds'])
            with (out/'round_metrics.jsonl').open('a',encoding='utf-8') as f:
                for r in rounds: f.write(json.dumps(r, default=str)+'\n')
            summary.update({'plan_id':pcfg['plan_id'],'seed':seed,'accuracy_loss':max(0, exp.get('reference_utility',1.0)-summary['validation_utility'])}); summaries.append(summary)
    pd.DataFrame(summaries).to_csv(out/'raw_metrics.csv', index=False)
if __name__=='__main__': main()
