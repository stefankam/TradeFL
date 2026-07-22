diff --git a/README.md b/README.md
index f10334c38a176b6ddb5a373208e0542011caaa5b..8b6175e431f749684e7640a637597f0487959f10 100644
--- a/README.md
+++ b/README.md
@@ -1,470 +1,133 @@
-# TokenFL: Unified Resource-Token Scheduling for Federated Edge Intelligence
+# TradeFL: Resource-Aware Federated LLM Fine-Tuning Plan Selection
 
-> **A unified operating system abstraction for resource-aware federated edge AI.**
+TradeFL is an experimental prototype for comparing federated LLM fine-tuning plans under common profiling conditions. It removes plans that violate hard constraints, normalizes cost-to-target measurements by deployment budgets, computes the weighted objective `J(P)`, and selects the lowest-cost feasible plan.
 
-TokenFL is a research prototype that introduces a **resource-token abstraction** for Federated Learning (FL) and Edge AI systems. Instead of optimizing memory, communication, computation, energy, latency, privacy, and storage independently, TokenFL models them as a unified resource budget and dynamically allocates them through a single optimization framework.
+## Installation
 
-The project is inspired by recent advances in federated learning, edge intelligence, and large language model (LLM) deployment, but proposes a fundamentally different abstraction layer: **every heterogeneous resource is represented using a common token currency**.
-
----
-
-# Motivation
-
-Modern Federated Learning and Edge AI systems optimize many resources independently.
-
-Existing research typically studies:
-
-* Client selection
-* Model partitioning
-* Memory optimization
-* Tensor management
-* Model compression
-* LoRA adaptation
-* Cloud-edge collaboration
-* Energy optimization
-* Runtime scheduling
-
-Although these approaches improve specific components, each introduces an independent optimization problem.
-
-For example,
-
-```
-Client Selection
-        ↓
-Memory Scheduler
-        ↓
-Split Scheduler
-        ↓
-Tensor Optimizer
-        ↓
-Energy Controller
-        ↓
-Cloud Scheduler
+```bash
+pip install -r requirements.txt
 ```
 
-These optimizers often compete with one another because they optimize different objectives.
-
-TokenFL proposes a different perspective.
-
-Instead of optimizing individual resources, every resource is represented as a **Resource Token**, allowing the runtime to reason about all system resources simultaneously.
+## Configuration
 
----
+The prototype is configured with YAML files:
 
-# Key Idea
+* `configs/experiment.yaml` defines shared experiment settings such as 5 clients, profiling rounds, seeds `[42, 43, 44]`, target quality, and hard constraints.
+* `configs/plans.yaml` enumerates concrete candidate configurations.
+* `configs/budgets.yaml` defines deployment budgets used for normalization and metric enablement.
+* `configs/weights.yaml` defines equal, memory-constrained, bandwidth-constrained, and quality-critical scenarios.
 
-Each edge device owns a fixed budget of **100 Resource Tokens**.
+## Supported plans
 
-Instead of directly allocating
+The first prototype exposes a common `TrainingPlan` interface and deterministic executable plan backends for:
 
-* GB of memory
-* GHz of CPU
-* Mbps bandwidth
-* Joules
-* milliseconds
+* `full_finetuning`
+* `lora_rank_8` and `lora_rank_16`
+* `qlora_4bit_rank_8`
+* `splitfed_layer_8`
+* `splitfed_layer_8_activation_compression`
+* `federated_distillation`
 
-TokenFL converts every resource into a common representation.
+The synthetic backends produce auditable memory, compute, communication, latency, privacy, and utility measurements so the complete selection pipeline can run before a real LLM training backend is attached.
 
-For example,
+## How to run profiling
 
+```bash
+python scripts/run_profile.py \
+  --config configs/experiment.yaml \
+  --plan lora_rank_8 \
+  --seed 42
 ```
-100 Resource Tokens
-
-Memory .............. 20
-Compute ............. 25
-Communication ....... 15
-Energy .............. 10
-Latency ............. 15
-Privacy ............. 5
-Storage ............. 5
-Accuracy Reserve .... 5
-```
-
-The runtime continuously redistributes tokens as system conditions change.
-
----
-
-# Resource Exchange
-
-The central idea of TokenFL is that every optimization becomes a **resource exchange**.
-
-Examples:
 
-### Checkpointing
+This appends per-round records to `outputs/round_metrics.jsonl` and cost-to-target summary records to `outputs/raw_metrics.csv`.
 
-```
-+10 Memory Tokens
+## How to run complete experiments
 
--4 Compute Tokens
+```bash
+python scripts/run_full_experiment.py --config configs/experiment.yaml
 ```
 
----
+This profiles all configured plans for all configured seeds using the same target quality, maximum rounds, client count, and random seed controls.
 
-### Compression
+## How budgets are defined
 
-```
-+15 Memory Tokens
+Budgets are stable deployment limits, not candidate-set min/max values. For example:
 
--2 Accuracy Tokens
+```text
+normalized_memory = peak_memory_bytes / memory_bytes_budget
+normalized_compute = compute_to_target_seconds / compute_seconds_budget
+normalized_communication = communication_to_target_bytes / communication_bytes_budget
+normalized_latency = latency_to_target_seconds / latency_seconds_budget
+normalized_privacy = privacy_risk / privacy_risk_budget
+normalized_accuracy = accuracy_loss / accuracy_loss_budget
 ```
 
----
+A normalized value below 1 is within budget, 1 is exactly at budget, and above 1 exceeds budget.
 
-### Cloud Offloading
+## How weights are defined
 
-```
-+10 Accuracy Tokens
+Weights are non-negative and must sum to 1 before disabled metrics are removed. When a metric such as energy is disabled in `configs/budgets.yaml`, TradeFL removes that metric and renormalizes the remaining enabled weights. The effective weights are recorded in `outputs/selection_results.json`.
 
--8 Communication Tokens
+## How feasibility is determined
 
--5 Privacy Tokens
-```
+Hard constraints are checked before scoring. The prototype checks peak memory, round latency, total time to target, minimum validation utility, privacy risk, required target attainment, edge infrastructure, and public-cloud policy. Weighted scoring never compensates for hard constraint violations.
 
----
+## How normalized costs and `J(P)` are calculated
 
-### DVFS
+For each feasible plan, TradeFL calculates:
 
+```text
+J(P) =
+    w_memory        * normalized_memory
+  + w_compute       * normalized_compute
+  + w_communication * normalized_communication
+  + w_energy        * normalized_energy
+  + w_latency       * normalized_latency
+  + w_privacy       * normalized_privacy
+  + w_accuracy      * normalized_accuracy_loss
 ```
-+8 Energy Tokens
 
--5 Latency Tokens
-```
+The selected plan is the feasible plan with the minimum score.
 
----
+## Plan selection
 
-### Model Splitting
-
-```
-+15 Memory Tokens
-
--10 Communication Tokens
+```bash
+python scripts/select_plan.py \
+  --results outputs/raw_metrics.csv \
+  --budgets configs/budgets.yaml \
+  --weights configs/weights.yaml \
+  --scenario equal
 ```
 
-Instead of having multiple schedulers, TokenFL optimizes all exchanges jointly.
-
----
+## Result summaries
 
-# Architecture
-
-```
-                    TokenFL Runtime
-
-                 ┌──────────────────────┐
-                 │ Resource Monitor     │
-                 └──────────┬───────────┘
-                            │
-                            ▼
-                 ┌──────────────────────┐
-                 │ Token Manager        │
-                 │ (100 Token Budget)   │
-                 └──────────┬───────────┘
-                            │
-                            ▼
-                 ┌──────────────────────┐
-                 │ Unified Optimizer    │
-                 └──────────┬───────────┘
-                            │
-        ┌──────────┬────────┼────────┬──────────┐
-        ▼          ▼        ▼        ▼          ▼
- Client      Split      Memory     LLM      Cloud
-Selection   Scheduler   Manager   Runtime   Scheduler
+```bash
+python scripts/summarize_results.py \
+  --input outputs/plan_summary.csv \
+  --output outputs/summary
 ```
 
-Unlike existing systems, these modules do **not** optimize independently.
-
-They become policy plugins controlled by one optimizer.
-
----
-
-# Policy Modules
-
-TokenFL currently implements five research-inspired policies.
-
-## 1. Client Selection Policy
-
-Inspired by modern FL client scheduling.
-
-Optimizes:
+## Output formats
 
-* communication
-* computation
-* data quality
+TradeFL writes machine-readable outputs:
 
----
+* `outputs/raw_metrics.csv` contains raw cost-to-target metrics.
+* `outputs/round_metrics.jsonl` stores incremental per-round metrics.
+* `outputs/plan_summary.csv` contains feasibility, normalized components, and scores.
+* `outputs/constraint_violations.csv` lists infeasible plans and violations.
+* `outputs/selection_results.json` contains the selected plan, score, normalized costs, effective weights, violated constraints, and nearest alternatives.
 
-## 2. Split Scheduling Policy
-
-Determines where models should be partitioned.
-
-Optimizes:
-
-* memory
-* communication
-* latency
-
----
-
-## 3. Memory Optimization Policy
-
-Chooses among
-
-* checkpointing
-* recomputation
-* compression
-* activation caching
-
-Optimizes:
-
-* memory
-* compute
-* accuracy
-
----
-
-## 4. Edge LLM Runtime Policy
-
-Selects
-
-* pruning
-* quantization
-* LoRA rank
-* DVFS level
-
-Optimizes
-
-* latency
-* energy
-* memory
-
----
-
-## 5. Cloud-Edge Collaboration Policy
-
-Determines
-
-* local inference
-* cloud inference
-* hybrid inference
-
-Optimizes
-
-* privacy
-* communication
-* latency
-* accuracy
-
----
-
-# Dynamic Runtime
-
-During execution TokenFL continuously observes
-
-* battery
-* CPU utilization
-* GPU utilization
-* memory pressure
-* bandwidth
-* network latency
-* thermal state
-* workload changes
-
-The token distribution is updated every scheduling round.
-
-Example
+## Minimal reproducible example
 
+```bash
+python scripts/run_full_experiment.py --config configs/experiment.yaml
+python scripts/select_plan.py --results outputs/raw_metrics.csv --budgets configs/budgets.yaml --weights configs/weights.yaml --scenario memory_constrained
+python scripts/select_plan.py --results outputs/raw_metrics.csv --budgets configs/budgets.yaml --weights configs/weights.yaml --scenario bandwidth_constrained
+python scripts/select_plan.py --results outputs/raw_metrics.csv --budgets configs/budgets.yaml --weights configs/weights.yaml --scenario quality_critical
 ```
-Round 1
-
-Memory        20
-Compute       25
-Communication 15
-Energy        15
-Latency       10
-Privacy        5
-Storage        5
-Accuracy       5
-```
-
-Battery becomes low.
-
-```
-Round 20
-
-Memory        18
-Compute       18
-Communication 12
-Energy        25
-Latency       10
-Privacy        7
-Storage        5
-Accuracy       5
-```
-
-The optimizer automatically adapts the execution strategy.
-
----
-
-# Repository Structure
-
-```
-tokenfl/
-
-    device.py
-    tokens.py
-    optimizer.py
-    simulator.py
-    runtime.py
-
-    policies/
-
-        client_selection.py
-        split_scheduler.py
-        memory_manager.py
-        llm_runtime.py
-        cloud_scheduler.py
-
-    evaluation.py
-    baselines.py
-    main.py
-
-tests/
-
-results/
-
-README.md
-requirements.txt
-```
-
----
-
-# Running
-
-Create a virtual environment
-
-```
-python -m venv venv
-source venv/bin/activate
-```
-
-Install dependencies
-
-```
-pip install -r requirements.txt
-```
-
-Run
-
-```
-python -m tokenfl.main \
-    --devices 50 \
-    --rounds 100 \
-    --seed 42
-```
-
----
-
-# Output
-
-After execution
-
-```
-results/
-
-metrics.csv
-
-execution_plan.json
-
-token_distribution.csv
-
-utility.png
-
-accuracy.png
-
-energy.png
-
-latency.png
-
-memory.png
-
-communication.png
-
-privacy.png
-```
-
----
-
-# Baselines
-
-The simulator includes
-
-* Random Scheduling
-* Fixed Client Selection
-* Static Model Split
-* Local-only Execution
-* Cloud-only Execution
-* Greedy Memory Allocation
-* Round-Robin Scheduling
-
----
-
-# Research Questions
-
-TokenFL is designed to investigate several research questions.
-
-### RQ1
-
-Can heterogeneous resources be represented by a unified token abstraction?
-
----
-
-### RQ2
-
-Can a single optimizer outperform specialized schedulers?
-
----
-
-### RQ3
-
-How should resource tokens be dynamically exchanged under runtime variance?
-
----
-
-### RQ4
-
-What is the optimal token allocation under changing edge environments?
-
----
-
-### RQ5
-
-How much utility is gained through unified optimization compared to independent optimization?
-
----
-
-# Potential Evaluation Metrics
-
-* Model Accuracy
-* Communication Cost
-* Memory Consumption
-* Energy Consumption
-* Average Latency
-* Device Participation
-* Successful Training Rounds
-* Resource Utilization
-* Token Allocation Stability
-* Utility Score
-
----
-
-# Long-Term Vision
-
-Current Edge AI systems resemble early operating systems, where each subsystem independently manages one hardware resource.
-
-TokenFL explores the possibility of a **Resource Operating System for AI**, where memory, computation, communication, energy, storage, privacy, and accuracy are treated as interchangeable resources managed through a unified optimization framework.
-
-Rather than introducing another scheduler, TokenFL proposes a new systems abstraction that could serve as a common runtime layer for future Federated Learning, Edge Intelligence, and Large Language Model deployment systems.
 
----
+The default configuration uses 5 clients, one small-transformer placeholder configuration, a classification-task utility target, full fine-tuning, LoRA, QLoRA, SplitFed variants, distillation, and three fixed random seeds.
 
-# License
+## Current limitations
 
-This repository is intended as a research prototype for academic use. Future extensions may integrate real federated learning frameworks (e.g., Flower, FedML, FedScale) and edge inference runtimes (e.g., ONNX Runtime, TensorRT, vLLM, MNN, ExecuTorch) to validate the token abstraction on practical deployments.
+This is the first experimental milestone. Plan implementations are deterministic synthetic executables, not real transformer training loops. Energy measurement exposes a pluggable interface but returns `None` by default when hardware telemetry is unavailable. Empirical privacy attacks are intentionally left as future extensions behind the privacy scoring interface.
