"""Auditable TradeFL plan selection."""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from .feasibility import Constraints, check_feasibility
from .normalization import Budgets, normalize_metrics
from .scoring import TradeFLWeights, calculate_score, renormalize_weights

def select_best(rows:list[dict], budgets:Budgets, weights:TradeFLWeights, constraints:Constraints, enabled:dict[str,bool]|None=None, reference_utility:float=1.0, nearest:int=3)->dict:
    enabled=enabled or {}; effective=renormalize_weights(weights, {k: enabled.get(k, True) for k in weights.asdict()})
    evaluated=[]
    for row in rows:
        row=dict(row); row['accuracy_loss']=row.get('accuracy_loss', max(0.0, reference_utility-float(row['validation_utility'])))
        feas=check_feasibility(row, constraints); row['feasible']=feas.feasible; row['violations']=';'.join(feas.violations)
        if feas.feasible:
            norm=normalize_metrics(row, budgets, enabled); score=calculate_score(norm, effective)
        else:
            norm={k:None for k in weights.asdict()}; score=None
        row.update({f'normalized_{k}':v for k,v in norm.items()}); row['tradefl_score']=score; evaluated.append(row)
    feasible=sorted([r for r in evaluated if r['feasible']], key=lambda r: (r['tradefl_score'], r['plan_id']))
    selected=feasible[0] if feasible else None
    return {'selected_plan': None if selected is None else selected['plan_id'], 'score': None if selected is None else selected['tradefl_score'],
            'normalized_costs': None if selected is None else {k:selected.get(f'normalized_{k}') for k in weights.asdict()},
            'effective_weights': effective.asdict(), 'violated_constraints': [] if selected else sorted({v for r in evaluated for v in str(r.get('violations','')).split(';') if v}),
            'alternatives': [{'plan_id':r['plan_id'],'score':r['tradefl_score']} for r in feasible[1:nearest+1]], 'evaluated_plans': evaluated}

def write_selection(result:dict, output_dir:str|Path='outputs')->None:
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result['evaluated_plans']).to_csv(out/'plan_summary.csv', index=False)
    pd.DataFrame([r for r in result['evaluated_plans'] if r.get('violations')]).to_csv(out/'constraint_violations.csv', index=False)
    slim={k:v for k,v in result.items() if k!='evaluated_plans'}
    (out/'selection_results.json').write_text(json.dumps(slim, indent=2), encoding='utf-8')
