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
