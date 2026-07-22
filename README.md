 # TradeFL: Resource-Aware Federated LLM Fine-Tuning Plan Selection
 
 TradeFL is an experimental prototype for comparing federated LLM fine-tuning plans under common profiling conditions. It removes plans that violate hard constraints, normalizes cost-to-target measurements by deployment budgets, computes the weighted objective `J(P)`, and selects the lowest-cost feasible plan.
 
 ## Installation
 
 ```bash
 pip install -r requirements.txt
 ```
 
 ## Configuration
 
 The prototype is configured with YAML files:
 
 * `configs/experiment.yaml` defines shared experiment settings such as 5 clients, profiling rounds, seeds `[42, 43, 44]`, target quality, and hard constraints.
 * `configs/plans.yaml` enumerates concrete candidate configurations.
 * `configs/budgets.yaml` defines deployment budgets used for normalization and metric enablement.
 * `configs/weights.yaml` defines equal, memory-constrained, bandwidth-constrained, and quality-critical scenarios.
 
 ## Supported plans
 
 The first prototype exposes a common `TrainingPlan` interface and deterministic executable plan backends for:
 
 * `full_finetuning`
 * `lora_rank_8` and `lora_rank_16`
 * `qlora_4bit_rank_8`
 * `splitfed_layer_8`
 * `splitfed_layer_8_activation_compression`
 * `federated_distillation`
 
 The synthetic backends produce auditable memory, compute, communication, latency, privacy, and utility measurements so the complete selection pipeline can run before a real LLM training backend is attached.
 
 ## How to run profiling
 
 ```bash
 python scripts/run_profile.py \
   --config configs/experiment.yaml \
   --plan lora_rank_8 \
   --seed 42
 ```
 
 This appends per-round records to `outputs/round_metrics.jsonl` and cost-to-target summary records to `outputs/raw_metrics.csv`.
 
 ## How to run complete experiments
 
 ```bash
 python scripts/run_full_experiment.py --config configs/experiment.yaml
 ```
 
 This profiles all configured plans for all configured seeds using the same target quality, maximum rounds, client count, and random seed controls.
 
 ## How budgets are defined
 
 Budgets are stable deployment limits, not candidate-set min/max values. For example:
 
 ```text
 normalized_memory = peak_memory_bytes / memory_bytes_budget
 normalized_compute = compute_to_target_seconds / compute_seconds_budget
 normalized_communication = communication_to_target_bytes / communication_bytes_budget
 normalized_latency = latency_to_target_seconds / latency_seconds_budget
 normalized_privacy = privacy_risk / privacy_risk_budget
 normalized_accuracy = accuracy_loss / accuracy_loss_budget
 ```
 
 A normalized value below 1 is within budget, 1 is exactly at budget, and above 1 exceeds budget.
 
 ## How weights are defined
 
 Weights are non-negative and must sum to 1 before disabled metrics are removed. When a metric such as energy is disabled in `configs/budgets.yaml`, TradeFL removes that metric and renormalizes the remaining enabled weights. The effective weights are recorded in `outputs/selection_results.json`.
 
 ## How feasibility is determined
 
 Hard constraints are checked before scoring. The prototype checks peak memory, round latency, total time to target, minimum validation utility, privacy risk, required target attainment, edge infrastructure, and public-cloud policy. Weighted scoring never compensates for hard constraint violations.
 
 ## How normalized costs and `J(P)` are calculated
 
 For each feasible plan, TradeFL calculates:
 
 ```text
 J(P) =
     w_memory        * normalized_memory
   + w_compute       * normalized_compute
   + w_communication * normalized_communication
   + w_energy        * normalized_energy
   + w_latency       * normalized_latency
   + w_privacy       * normalized_privacy
   + w_accuracy      * normalized_accuracy_loss
 ```
 
 The selected plan is the feasible plan with the minimum score.
 
 ## Plan selection
 
 ```bash
 python scripts/select_plan.py \
   --results outputs/raw_metrics.csv \
   --budgets configs/budgets.yaml \
   --weights configs/weights.yaml \
   --scenario equal
 ```
 
 ## Result summaries
 
 ```bash
 python scripts/summarize_results.py \
   --input outputs/plan_summary.csv \
   --output outputs/summary
 ```
 
 ## Output formats
 
 TradeFL writes machine-readable outputs:
 
 * `outputs/raw_metrics.csv` contains raw cost-to-target metrics.
 * `outputs/round_metrics.jsonl` stores incremental per-round metrics.
 * `outputs/plan_summary.csv` contains feasibility, normalized components, and scores.
 * `outputs/constraint_violations.csv` lists infeasible plans and violations.
 * `outputs/selection_results.json` contains the selected plan, score, normalized costs, effective weights, violated constraints, and nearest alternatives.
 
 ## Minimal reproducible example
 
 ```bash
 python scripts/run_full_experiment.py --config configs/experiment.yaml
 python scripts/select_plan.py --results outputs/raw_metrics.csv --budgets configs/budgets.yaml --weights configs/weights.yaml --scenario memory_constrained
 python scripts/select_plan.py --results outputs/raw_metrics.csv --budgets configs/budgets.yaml --weights configs/weights.yaml --scenario bandwidth_constrained
 python scripts/select_plan.py --results outputs/raw_metrics.csv --budgets configs/budgets.yaml --weights configs/weights.yaml --scenario quality_critical
 ```
 
 The default configuration uses 5 clients, one small-transformer placeholder configuration, a classification-task utility target, full fine-tuning, LoRA, QLoRA, SplitFed variants, distillation, and three fixed random seeds.
 
 ## Current limitations
 
 This is the first experimental milestone. Plan implementations are deterministic synthetic executables, not real transformer training loops. Energy measurement exposes a pluggable interface but returns `None` by default when hardware telemetry is unavailable. Empirical privacy attacks are intentionally left as future extensions behind the privacy scoring interface.
