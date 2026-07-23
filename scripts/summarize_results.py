#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    df=pd.read_csv(args.input); out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    summary=df.groupby('plan_id', as_index=False).agg({'tradefl_score':'mean','peak_memory_bytes':'mean','communication_to_target_bytes':'mean','validation_utility':'mean','feasible':'max'})
    summary.to_csv(str(out)+'.csv', index=False)
if __name__=='__main__': main()
