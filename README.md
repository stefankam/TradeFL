# TokenFL: Unified Resource-Token Scheduling for Federated Edge Intelligence

> **A unified operating system abstraction for resource-aware federated edge AI.**

TokenFL is a research prototype that introduces a **resource-token abstraction** for Federated Learning (FL) and Edge AI systems. Instead of optimizing memory, communication, computation, energy, latency, privacy, and storage independently, TokenFL models them as a unified resource budget and dynamically allocates them through a single optimization framework.

The project is inspired by recent advances in federated learning, edge intelligence, and large language model (LLM) deployment, but proposes a fundamentally different abstraction layer: **every heterogeneous resource is represented using a common token currency**.

---

# Motivation

Modern Federated Learning and Edge AI systems optimize many resources independently.

Existing research typically studies:

* Client selection
* Model partitioning
* Memory optimization
* Tensor management
* Model compression
* LoRA adaptation
* Cloud-edge collaboration
* Energy optimization
* Runtime scheduling

Although these approaches improve specific components, each introduces an independent optimization problem.

For example,

```
Client Selection
        ↓
Memory Scheduler
        ↓
Split Scheduler
        ↓
Tensor Optimizer
        ↓
Energy Controller
        ↓
Cloud Scheduler
```

These optimizers often compete with one another because they optimize different objectives.

TokenFL proposes a different perspective.

Instead of optimizing individual resources, every resource is represented as a **Resource Token**, allowing the runtime to reason about all system resources simultaneously.

---

# Key Idea

Each edge device owns a fixed budget of **100 Resource Tokens**.

Instead of directly allocating

* GB of memory
* GHz of CPU
* Mbps bandwidth
* Joules
* milliseconds

TokenFL converts every resource into a common representation.

For example,

```
100 Resource Tokens

Memory .............. 20
Compute ............. 25
Communication ....... 15
Energy .............. 10
Latency ............. 15
Privacy ............. 5
Storage ............. 5
Accuracy Reserve .... 5
```

The runtime continuously redistributes tokens as system conditions change.

---

# Resource Exchange

The central idea of TokenFL is that every optimization becomes a **resource exchange**.

Examples:

### Checkpointing

```
+10 Memory Tokens

-4 Compute Tokens
```

---

### Compression

```
+15 Memory Tokens

-2 Accuracy Tokens
```

---

### Cloud Offloading

```
+10 Accuracy Tokens

-8 Communication Tokens

-5 Privacy Tokens
```

---

### DVFS

```
+8 Energy Tokens

-5 Latency Tokens
```

---

### Model Splitting

```
+15 Memory Tokens

-10 Communication Tokens
```

Instead of having multiple schedulers, TokenFL optimizes all exchanges jointly.

---

# Architecture

```
                    TokenFL Runtime

                 ┌──────────────────────┐
                 │ Resource Monitor     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Token Manager        │
                 │ (100 Token Budget)   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Unified Optimizer    │
                 └──────────┬───────────┘
                            │
        ┌──────────┬────────┼────────┬──────────┐
        ▼          ▼        ▼        ▼          ▼
 Client      Split      Memory     LLM      Cloud
Selection   Scheduler   Manager   Runtime   Scheduler
```

Unlike existing systems, these modules do **not** optimize independently.

They become policy plugins controlled by one optimizer.

---

# Policy Modules

TokenFL currently implements five research-inspired policies.

## 1. Client Selection Policy

Inspired by modern FL client scheduling.

Optimizes:

* communication
* computation
* data quality

---

## 2. Split Scheduling Policy

Determines where models should be partitioned.

Optimizes:

* memory
* communication
* latency

---

## 3. Memory Optimization Policy

Chooses among

* checkpointing
* recomputation
* compression
* activation caching

Optimizes:

* memory
* compute
* accuracy

---

## 4. Edge LLM Runtime Policy

Selects

* pruning
* quantization
* LoRA rank
* DVFS level

Optimizes

* latency
* energy
* memory

---

## 5. Cloud-Edge Collaboration Policy

Determines

* local inference
* cloud inference
* hybrid inference

Optimizes

* privacy
* communication
* latency
* accuracy

---

# Dynamic Runtime

During execution TokenFL continuously observes

* battery
* CPU utilization
* GPU utilization
* memory pressure
* bandwidth
* network latency
* thermal state
* workload changes

The token distribution is updated every scheduling round.

Example

```
Round 1

Memory        20
Compute       25
Communication 15
Energy        15
Latency       10
Privacy        5
Storage        5
Accuracy       5
```

Battery becomes low.

```
Round 20

Memory        18
Compute       18
Communication 12
Energy        25
Latency       10
Privacy        7
Storage        5
Accuracy       5
```

The optimizer automatically adapts the execution strategy.

---

# Repository Structure

```
tokenfl/

    device.py
    tokens.py
    optimizer.py
    simulator.py
    runtime.py

    policies/

        client_selection.py
        split_scheduler.py
        memory_manager.py
        llm_runtime.py
        cloud_scheduler.py

    evaluation.py
    baselines.py
    main.py

tests/

results/

README.md
requirements.txt
```

---

# Running

Create a virtual environment

```
python -m venv venv
source venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

Run

```
python -m tokenfl.main \
    --devices 50 \
    --rounds 100 \
    --seed 42
```

---

# Output

After execution

```
results/

metrics.csv

execution_plan.json

token_distribution.csv

utility.png

accuracy.png

energy.png

latency.png

memory.png

communication.png

privacy.png
```

---

# Baselines

The simulator includes

* Random Scheduling
* Fixed Client Selection
* Static Model Split
* Local-only Execution
* Cloud-only Execution
* Greedy Memory Allocation
* Round-Robin Scheduling

---

# Research Questions

TokenFL is designed to investigate several research questions.

### RQ1

Can heterogeneous resources be represented by a unified token abstraction?

---

### RQ2

Can a single optimizer outperform specialized schedulers?

---

### RQ3

How should resource tokens be dynamically exchanged under runtime variance?

---

### RQ4

What is the optimal token allocation under changing edge environments?

---

### RQ5

How much utility is gained through unified optimization compared to independent optimization?

---

# Potential Evaluation Metrics

* Model Accuracy
* Communication Cost
* Memory Consumption
* Energy Consumption
* Average Latency
* Device Participation
* Successful Training Rounds
* Resource Utilization
* Token Allocation Stability
* Utility Score

---

# Long-Term Vision

Current Edge AI systems resemble early operating systems, where each subsystem independently manages one hardware resource.

TokenFL explores the possibility of a **Resource Operating System for AI**, where memory, computation, communication, energy, storage, privacy, and accuracy are treated as interchangeable resources managed through a unified optimization framework.

Rather than introducing another scheduler, TokenFL proposes a new systems abstraction that could serve as a common runtime layer for future Federated Learning, Edge Intelligence, and Large Language Model deployment systems.

---

# License

This repository is intended as a research prototype for academic use. Future extensions may integrate real federated learning frameworks (e.g., Flower, FedML, FedScale) and edge inference runtimes (e.g., ONNX Runtime, TensorRT, vLLM, MNN, ExecuTorch) to validate the token abstraction on practical deployments.
