# Plan for real federated LLM fine-tuning

## Supported roles

The requested models cannot all be averaged in one federation. FedAvg combines
matching parameters, so LLaMA 3, DeepSeek-R1-Distill-Llama, and
Bio_ClinicalBERT must be run as **three separate federated experiments** and
compared afterward:

1. `meta-llama/Meta-Llama-3-8B` with PEFT LoRA;
2. `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` with PEFT LoRA; and
3. `emilyalsentzer/Bio_ClinicalBERT` as the sequence-classification baseline.

`gpt-4o` is an API-hosted model whose parameters are not available to this
process. It cannot produce local adapter tensors for FedAvg and must not be
configured as a federated participant. It can be an external teacher (generate
labels/rationales before local training) or an evaluator. Provider-managed API
fine-tuning, where available for a model/account, is still not client-local
FedAvg.


An opt-in centralized GPT-4o baseline is implemented with
`--run-external-baselines`. It evaluates the common validation and test splits
through the provider API but never treats API responses as federated model
updates. This option can incur API charges and sends configured dataset text to
an external service, so it is disabled by default and requires `OPENAI_API_KEY`.

GPU board energy is measured by periodically sampling `nvidia-smi` power draw
and integrating watts over each complete federated round. Energy remains null
when NVIDIA telemetry is unavailable; API-provider energy is not observable and
is also reported as unavailable rather than estimated.

Live reporting begins after the first completed federated round. The runner
atomically replaces `raw_metrics.csv`, refreshes selection CSV/JSON artifacts,
and regenerates the summary CSV and PDF graph set after every later round. The
current in-progress seed is represented by its latest completed round, so an
interrupted long run still leaves inspectable tables and graphs. Graph rendering
adds some per-round overhead but never runs concurrently with model training.

To evaluate GPT-4o without starting any local federated model, pass
`--external-only` to `run_real_federated_pipeline.py`. This mode implies external
baseline execution and rejects combinations with `--experiment-id` or
`--cpu-smoke-test` so an expensive local run cannot start accidentally.

The catalog also provides `qwen2_5_7b_centralized_baseline`, which evaluates
`Qwen/Qwen2.5-7B-Instruct` locally through Transformers without an API key or
per-request provider charge. Use `--external-only --external-baseline-id
qwen2_5_7b_centralized_baseline` to select it without calling GPT-4o. The model
still consumes local accelerator time, electricity, storage, and download
bandwidth; users remain responsible for reviewing its model license and data
governance requirements.

When federated outputs already exist, add `--append-results` together with
`--external-only` and one or more `--external-baseline-id` filters. The runner
loads the existing `raw_metrics.csv`, preserves `round_metrics.jsonl` and skipped
records, replaces any older row for the selected baseline, then regenerates
selection artifacts and graphs from the merged results. Append mode refuses to
run without an existing raw metrics file or without `--external-only`.

`summarize_results.py` also performs an offline scheduler replay whenever its
input contains the measured utility, memory, and latency columns. It compares
shadow-price TradeFL, independent bottleneck minimization, a static weighted sum,
greedy utility, fixed-price TradeFL, and a hindsight enumerated oracle. The
replay writes decision and definition CSVs plus graphs for utility, SLO and
resource violations, communication, energy, time to target, scheduling
overhead, and target attainment. These are counterfactual selections over
already observed plan/seed rows—not new model training and not a claim about
unobserved workload arrivals or deadlines.

In replay, dynamic TradeFL initializes each constrained resource price to 1.0
and applies a projected subgradient update after every seed: `lambda <-
max(0, lambda + 0.25 * (selected_usage / capacity - 1))`. Memory, round and
total latency, privacy, and the minimum-quality floor receive prices when their
constraints are configured. The fixed-price ablation uses the identical
Lagrangian score but keeps every initial price at 1.0. The generated
`scheduler_shadow_price_trace.csv` records every before/after price, ratio, and
subgradient so the comparison is auditable.

All scheduling policy implementations are kept under `tradefl/scheduling`.
`replay.py` owns the dynamic TradeFL, fixed-price, independent/per-resource,
static weighted-sum, greedy, and oracle selectors plus replay outcome
accounting. `shadow_pricing.py` owns the online training-loop scheduler.
`scripts/summarize_results.py` only persists replay tables and renders graphs;
it does not implement a scheduling policy.

## Online scheduling treatments and auditable evaluation

The real runner now uses the common `OnlineScheduler` interface for five
training-time treatments: dynamic-price TradeFL, fixed-price TradeFL,
independent/per-resource, static weighted-sum, and greedy utility. Every policy
receives the identical enumerated client-subset action set, the same resource
availability configuration, the same client partitions, and the same seed.
Configured treatments produce distinct plan IDs such as
`bio_clinicalbert_baseline__tradefl_dynamic` and
`bio_clinicalbert_baseline__tradefl_fixed`; they are separate training runs,
not labels applied to one trained model. Use repeatable `--scheduler-policy`
arguments to run only selected configured treatments.

Each `(model, policy, seed)` constructs a fresh scheduler, so dynamic prices
reset to their configured initial values at every independent seed. The runner
writes every chosen client-subset action to `online_scheduler_actions.jsonl`
and every before/after price vector, predicted demand, realized demand, and
availability vector to `online_shadow_price_trajectory.jsonl`. Round and
run-summary rows also carry `target_quality` and serialized constraint
provenance.

Offline comparison uses a multi-task trace within each seed. If input rows
provide `workload_task_id`, those arrivals are used directly; otherwise the
common measured candidate set is replayed as three ordered task arrivals. All
policies see the same candidate set for a task, and dynamic replay prices reset
between seeds but update between tasks. The hindsight Oracle exhaustively
enumerates the common feasible actions, prioritizing target attainment, then
test utility, then latency. It is explicitly not an online policy.

Scheduler reports include `scheduler_comparison_per_seed.csv`, 95% confidence
interval columns in `scheduler_comparison.csv`, and visible per-seed points on
the aggregate graphs. Successful time-to-target and unsuccessful censored
observation horizons are written and plotted as separate metrics rather than
being combined into one misleading bar.

Shadow pricing is also part of the **actual federated training loop** when
`experiment.shadow_pricing.enabled` is true. Before each round, every subset of
`clients_per_round` clients is a candidate action. The scheduler predicts each
candidate's peak-memory, compute-time, and communication demand from per-client
exponential moving averages, evaluates `K(x) = sum_r p[r] * d[r](x)`, and runs
the minimum-cost candidate. It explores clients without measurements before it
uses the learned costs, preventing missing telemetry from permanently excluding
a client. After those clients really train, the scheduler reconciles their
measured demand against configured availability and applies
`p <- min(p_max, max(0, p + eta * (D - A)))`, with
`eta = learning_rate / A^2`. Thus the configured learning rate operates on a
dimensionless utilization violation while the implementation remains the
paper's raw-demand dual update. Peak memory uses a `max` reducer because clients
run sequentially; time and bytes use `sum`. Every round stores the candidate
count, reservation, token cost, availability, realized demand, and prices before
and after the update in `round_metrics.jsonl`.

Prices are resource-specific: the checked-in initial price is `1/A_r` and the
cap is `10/A_r`, so one full capacity unit initially costs one token and byte
counts cannot numerically swamp seconds merely because they use a larger unit.
These remain prices multiplying raw demand—not normalized demands—and can be
overridden independently for each physical resource.

For a training-time fixed-price ablation, copy the same experiment settings and
set `shadow_pricing.update_prices` to `false`; selection still uses the token
cost, but all prices remain at their configured initial value. Setting
`shadow_pricing.enabled` to `false` restores seeded random client sampling.


The model-role declaration is in `configs/real_federated_models.yaml`, and
`scripts/validate_real_federated_config.py` prevents an API model from being
silently treated as a trainable FedAvg participant.

## Required implementation phases

The checked-in bag-of-words pipeline is not sufficient for this experiment.
Real training requires the following work before results are produced:

### 1. Reproducible task definition

Define a common PubMedQA input template and label space. Causal LMs should be
trained to emit a constrained `yes`/`no`/`maybe` answer; Bio_ClinicalBERT should
use a three-class classification head. Use the same patient-safe splits and the
same evaluation examples for all three experiments.

### 2. Client partitioning

Partition only the training split into stable client datasets. Preserve
validation and test sets centrally for comparison. Record the partition seed,
client example counts, and label distribution. Select clients each round using
`client_sampling_ratio`; do not reuse the current evaluation-only five-slice
calculation as training clients.

### 3. Local model training

Use PyTorch, Transformers, Datasets, Accelerate, and PEFT. Each selected client
must receive the current global model/adapter, train only on its own records for
`local_epochs`, and return model deltas plus its number of examples. Use mixed
precision, gradient accumulation, checkpointing, and 4-bit loading when needed
to fit the 8B models. Authentication and license acceptance are required for
gated model repositories.

### 4. Aggregation

For LoRA experiments, aggregate only identically named adapter tensors with a
sample-count-weighted FedAvg. For Bio_ClinicalBERT full fine-tuning, aggregate
all matching trainable tensors. Never aggregate tensors across different base
models or architectures. Persist the global state and a round manifest so runs
can be resumed and audited.

### 5. Evaluation and accounting

After aggregation, evaluate the global model on the complete validation set,
apply early stopping, and evaluate the test set only according to the final
protocol. Count actual serialized tensor bytes, peak accelerator memory, local
training time, aggregation time, and end-to-end round latency. Do not reuse the
current pickled metadata payload size as model communication.

### 6. GPT-4o integration

If GPT-4o is used as a teacher, generate and cache a versioned JSONL artifact
before federated training. Store prompt version, model identifier, response,
source-example ID, and review status. Do not send protected health information
unless the deployment, contracts, consent, and data governance explicitly allow
it. If it is used as a judge, keep deterministic task metrics such as accuracy
and macro-F1 primary and report judge results separately.

## Hardware expectations

An 8B model cannot be realistically trained by the repository's current
dependency-light environment. Each simulated client needs accelerator capacity;
even with LoRA and low-bit loading, memory depends on sequence length, batch
size, optimizer, precision, and whether clients execute concurrently. Start with
one client process at a time, verify one local update, then two-client FedAvg,
and only then scale to the configured client count.

## Acceptance criteria for calling the result "real federated fine-tuning"

* Training data is physically partitioned and a client accesses only its split.
* At least two clients independently train model or adapter parameters.
* The server performs sample-weighted aggregation of real tensors.
* The next round starts from the aggregated global state.
* Communication metrics measure serialized tensors.
* `num_clients`, client sampling, local epochs, and aggregation config change
  runtime behavior and are covered by integration tests.
* Checkpoints and manifests identify the exact base model revision, tokenizer,
  adapter config, dataset manifest, seed, and software versions.

Until these criteria are met, output from the existing runner remains a
bag-of-words reference experiment rather than federated LLM fine-tuning.

## What is implemented now, and what is not

`tradefl/federation/fedavg.py` now contains two real, model-independent
federation primitives: deterministic disjoint client partitioning and
sample-count-weighted averaging of matching tensors. It rejects one-client
"federation" and incompatible tensor names or shapes.

`tradefl/federation/huggingface.py` implements real Hugging Face model and
tokenizer loading, PubMedQA formatting, PEFT LoRA/QLoRA setup, client-local
optimizer steps through `Trainer`, adapter extraction/reload, classifier and
causal-LM evaluation, accelerator peak-memory measurement, and tensor-byte
accounting. `scripts/run_real_federated_experiment.py` connects it to client
sampling and FedAvg and writes real-federated round and summary records.

`tradefl/backends/bow.py` remains only because fast unit and selection-pipeline
tests need a model that runs without PyTorch, Transformers, model credentials,
or a GPU. It is not a fallback for real experiments, its output is labeled
`reference_only`, its directory is ignored, and graph generation rejects it.
It remains isolated from the real runner and graph path.

## Running the complete experiment

Install the separate accelerator-backed environment and authenticate with the
model registry if the chosen checkpoint is gated:

```bash
python -m pip install -r requirements-federated.txt
huggingface-cli login
```

Run one model first (recommended for hardware validation):

```bash
python scripts/run_real_federated_experiment.py \
  --experiment-config configs/experiment_pubmedqa_real.yaml \
  --models-config configs/real_federated_models.yaml \
  --experiment-id bio_clinicalbert_baseline
```

Remove `--experiment-id` to run all three independent model experiments. Then
apply feasibility/scoring and generate graphs:

```bash
python scripts/select_plan.py \
  --results outputs/real_federated/raw_metrics.csv \
  --budgets configs/budgets.yaml \
  --weights configs/weights.yaml \
  --config configs/experiment_pubmedqa_real.yaml

python scripts/summarize_results.py \
  --input outputs/real_federated/plan_summary.csv \
  --output outputs/real_federated/summary
```

The 8B experiments require substantial accelerator memory and model downloads;
CI validates orchestration with test doubles but does not download or train the
checkpoints.

## Graph generation

The existing graph generator is reusable: it consumes metric columns rather
than model objects and writes 12 PDF graphs plus a CSV summary and graph
manifest. Genuine federated runners must add `training_mode=real_federated` to
every summary row and populate the same cost/utility columns expected by
`scripts/summarize_results.py`. Then run:

```bash
python scripts/summarize_results.py \
  --input outputs/real_federated/plan_summary.csv \
  --output outputs/real_federated/summary
```

The provenance guard is mandatory and deliberately rejects reference output;
there is no option to generate graphs from the bag-of-words runner.
