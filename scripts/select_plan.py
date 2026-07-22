#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import pandas as pd
from tradefl.selection.feasibility import Constraints
from tradefl.selection.normalization import Budgets
from tradefl.selection.scoring import TradeFLWeights
from tradefl.selection.selector import select_best, write_selection
from tradefl.utils.config import load_yaml

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',required=True); ap.add_argument('--budgets',required=True); ap.add_argument('--weights',required=True); ap.add_argument('--scenario'); args=ap.parse_args()
    rows=pd.read_csv(args.results).to_dict('records'); bcfg=load_yaml(args.budgets); wcfg=load_yaml(args.weights); ecfg=load_yaml('configs/experiment.yaml')
    scenario=args.scenario or wcfg.get('default_scenario','equal')
    result=select_best(rows, Budgets(**bcfg['budgets']), TradeFLWeights(**wcfg['scenarios'][scenario]), Constraints(**ecfg['constraints']), bcfg.get('enabled_metrics',{}), ecfg['experiment'].get('reference_utility',1.0))
    write_selection(result, ecfg['experiment'].get('output_dir','outputs'))
if __name__=='__main__': main()
