# Training architecture and current federated-learning status

## Short answer

TradeFL currently fits a dependency-light multinomial Naive Bayes text
classifier on real PubMedQA or MedNLI records. It does **not** load a pretrained
LLM, use gradients or an optimizer, or fine-tune Transformer weights. The plan
implementations are bag-of-words reference analogues used to exercise the
measurement and selection pipeline.

Despite the `tradefl/federation` package name and federated settings in the
experiment files, the current training path is **centralized, not federated
learning**. It does not partition training data among clients, select clients,
create client-local models, or perform FedAvg. `num_clients`,
`client_sampling_ratio`, and `aggregation` are presently declarative config
values that are not consumed by the training runner. The five validation slices
used for worst-client utility are evaluation-only slices; they are not training
clients.

## End-to-end execution path

1. `scripts/run_full_experiment.py` reads the experiment and plan configs.
2. `scripts/obtain_data.py` ensures the configured dataset splits exist.
   PubMedQA's original PQA-L data is deterministically shuffled and split
   80/10/10 into train, validation, and test files.
3. `tradefl/data/loaders.py` normalizes every source row into a
   `DatasetRecord(text, label, metadata)` and returns a `DatasetBundle`.
4. For each seed and configured plan, the runner imports
   `tradefl.plans.<type>` and calls its `build()` function.
5. `tradefl/federation/simulator.py::run_to_target` calls `setup()`, executes
   rounds until validation utility reaches the target or `max_rounds` is
   exhausted, evaluates the final model, and aggregates round costs.
6. In each round, `DatasetBackedFineTuningPlan` traverses the complete training
   split once for every `local_epochs`, divided into non-overlapping batches.
   It then evaluates the model on the complete validation and test splits.
7. The runner writes per-round JSONL records and one CSV summary per
   `(seed, plan)` run. Selection is a separate step performed by
   `scripts/select_plan.py`.

## Model and "fine-tuning" mechanics

`BagOfWordsFineTuningBackend` is both the reference model and the common backend
API. Its learned state consists of:

* a count of training examples for each label;
* token-frequency counts for each label; and
* the observed vocabulary.

Training tokenizes each record and adds its token and class counts. Prediction
uses a Laplace-smoothed multinomial Naive Bayes score:

```text
score(label, text) = log P(label) + sum(log P(token | label))
```

The label with the greatest score is returned. Evaluation computes accuracy and
macro-F1. Calling this process "fine-tuning" means incrementally fitting this
reference classifier; there is no pretrained checkpoint, neural network,
backpropagation, loss function, GPU training, tokenizer model, or optimizer.
Each plan is reset to an empty model during `setup()`.

## What each plan does

| Plan | Current executable behavior |
| --- | --- |
| Full fine-tuning | Fits the unrestricted bag-of-words Naive Bayes model. It is supported by its module but is not listed in the current `configs/plans.yaml`. |
| LoRA | Hashes tokens into `rank * 32` bounded adapter feature buckets, then fits Naive Bayes counts over those features. This is a reference analogue, not matrix low-rank adaptation. |
| QLoRA | Uses the same bounded adapter and scales serialized update-size accounting according to `quantization_bits`. It does not quantize neural weights. |
| SplitFed | Prefixes features as client- or server-side according to `split_layer`; activation compression drops alternating features. It does not execute separate client/server neural networks or exchange activations over a network. |
| Distillation | Fits a full teacher on the training split, uses the teacher's hard predictions as pseudo-labels, and fits a compact 64-bucket student. It does not use temperature-scaled soft logits. |

These strategies now take different code paths, but none is a faithful LLM
implementation. Their results should be described as reference-pipeline results,
not LoRA/QLoRA/SplitFed LLM benchmarks.

## Measurements and outputs

During a round, the plan measures wall-clock training time and process RSS,
evaluates validation/test utility, and constructs upload/download accounting
payloads. The communication counter measures the pickled size of those payloads;
it does not transmit data between real machines. Latency is modeled as compute
time plus payload bytes divided by a fixed 10 MB/s rate. Energy is currently
unavailable and is recorded as `None`.

The principal outputs are:

* `outputs/round_metrics.jsonl`: one record per executed round;
* `outputs/raw_metrics.csv`: one aggregate record per seed and plan;
* `outputs/plan_summary.csv`: raw records plus feasibility, normalization, and
  scoring fields;
* `outputs/constraint_violations.csv`: infeasible plan/seed summaries; and
* `outputs/selection_results.json`: selected plan and alternatives.

## What is required for actual federated LLM fine-tuning

An actual implementation still needs a pretrained Transformer and tokenizer,
PEFT/LoRA modules, an optimizer and gradient loop, deterministic client data
partitioning, client sampling, independent client-local updates, FedAvg (or
another server aggregation algorithm), real adapter/activation serialization,
and optional distributed transport. Until those components exist,
`num_clients: 5` and `aggregation: FedAvg` must not be interpreted as evidence
that the checked-in experiment performed federated learning.

## File map

| File | Responsibility |
| --- | --- |
| `configs/experiment*.yaml` | Dataset paths, seeds, round limits, targets, and currently declarative federation settings. |
| `configs/plans.yaml` | Candidate plan IDs and plan-specific parameters. |
| `scripts/obtain_data.py` | Obtains and splits PubMedQA or prepares MedNLI inputs. |
| `tradefl/data/loaders.py` | Validates and normalizes dataset records. |
| `scripts/run_full_experiment.py` | Orchestrates every seed/plan run and writes raw outputs. |
| `tradefl/plans/base.py` | Shared setup, complete-split batching, evaluation, and measurement loop. |
| `tradefl/plans/*.py` | Concrete reference strategy selection and overrides. |
| `tradefl/backends/bow.py` | Naive Bayes model, hashed adapter, and split-feature implementations. |
| `tradefl/federation/simulator.py` | Early-stopping round loop and cost-to-target aggregation; currently not a multi-client simulator. |
| `tradefl/selection/*.py` | Feasibility, normalization, scoring, and plan selection after training. |
