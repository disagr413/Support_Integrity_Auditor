# Support Integrity Auditor (SIA)

## Overview

Support Integrity Auditor (SIA) is an AI-powered system designed to detect priority mismatches in customer support tickets.

In large-scale support operations, tickets are often incorrectly prioritized:

* Critical issues may be assigned Low or Medium priority (Hidden Crisis)
* Routine requests may be escalated unnecessarily (False Alarm)

Such inconsistencies lead to:

* Delayed resolution of urgent cases
* Resource misallocation
* Increased customer dissatisfaction
* Operational inefficiencies

SIA automatically audits ticket prioritization decisions by combining:

1. Multi-signal pseudo-label generation
2. Semantic ticket understanding
3. DeBERTa-v3 + LoRA fine-tuning
4. Explainable evidence generation
5. Interactive audit dashboard

The system identifies potentially misclassified tickets and generates detailed evidence-backed explanations for auditors.

---

## Live Demo

Streamlit Application:

[https://supportintegrityauditor-lbwzos36mghad8udekrric.streamlit.app/](https://supportintegrityauditor-lbwzos36mghad8udekrric.streamlit.app/)

---

# Problem Statement

Customer support teams process thousands of tickets daily.

Traditional workflows rely heavily on manually assigned priorities:

* Low
* Medium
* High
* Critical

However, assigned priority often does not reflect actual severity.

Examples:

### Hidden Crisis

Assigned Priority: Low

Ticket Content:

> My account was hacked and unauthorized transactions were made.

Actual Severity:

Critical

---

### False Alarm

Assigned Priority: Critical

Ticket Content:

> Where is your headquarters located?

Actual Severity:

Low

---

The objective is to automatically identify such mismatches and provide explainable evidence.

---

# System Architecture

## Stage 1 — Pseudo-Label Generation

The system first creates high-quality pseudo-labels using multiple independent signals.

### Signal 1: NLP Keyword Severity

Detects indicators such as:

* Fraud
* Phishing
* Malware
* Data Breach
* Unauthorized Access
* Ransomware
* System Crash

Produces:

```text
sig_kw
```

### Signal 2: Resolution Time Mismatch

Compares actual resolution time against expected category-priority benchmarks.

Produces:

```text
sig_rt
```

### Signal 3: Lexical + Satisfaction Analysis

Uses:

* Severity vocabulary
* Routine vocabulary
* Customer satisfaction scores

Produces:

```text
sig_lex
```

### Signal 4: Semantic Clustering

Embeddings generated using:

```text
all-MiniLM-L6-v2
```

Then clustered using:

* HDBSCAN
* K-Means fallback

Produces:

```text
sig_sem
```

---

## Signal Fusion Layer

All signals are combined using:

```text
Logistic Regression Fusion
```

Output:

```text
sev_score
```

which represents inferred severity.

Severity mapping:

| Score Range | Severity |
| ----------- | -------- |
| < 0.25      | Low      |
| 0.25 – 0.50 | Medium   |
| 0.50 – 0.75 | High     |
| > 0.75      | Critical |

---

## Mismatch Detection

The system compares:

```text
Assigned Priority
vs
Inferred Severity
```

Classification:

### Consistent

Assigned priority matches inferred severity.

### Hidden Crisis

Actual severity is significantly higher than assigned priority.

### False Alarm

Assigned priority is significantly higher than actual severity.

---

# Stage 2 — DeBERTa-v3 + LoRA

The generated pseudo-labels are used to train a lightweight transformer classifier.

Base Model:

```text
microsoft/deberta-v3-small
```

Fine-Tuning:

```text
LoRA (Low Rank Adaptation)
```

Advantages:

* Faster training
* Lower memory usage
* Efficient deployment
* Minimal trainable parameters

Training includes:

* AdamW optimizer
* Cosine learning rate scheduling
* Class balancing
* Threshold optimization
* Validation-based model selection

---

# Explainability Engine

For every flagged ticket, SIA generates an Evidence Dossier.

The dossier contains:

### Severity Assessment

* Assigned Priority
* Inferred Severity
* Confidence Score

### Keyword Evidence

Examples:

* fraud
* phishing
* hacked
* unauthorized access
* ransomware

### Resolution Time Evidence

Compares actual resolution duration against expected benchmarks.

### Satisfaction Evidence

Uses customer satisfaction signals to strengthen severity inference.

### Category Baseline Analysis

Evaluates whether assigned priority aligns with category-specific severity expectations.

### Constraint Analysis

Generates a human-readable explanation describing:

* Why the ticket was flagged
* Supporting evidence
* Operational implications

---

# Dashboard Features

## Upload Support Tickets

Upload CSV files containing support ticket records.

---

## Priority Mismatch Detection

Automatically identify:

* Hidden Crises
* False Alarms
* Consistent Tickets

---

## Confidence Scores

Displays model confidence for every prediction.

---

## Interactive Analytics

Visualizations include:

* Priority Distribution
* Category Distribution
* Mismatch Breakdown
* Confidence Analysis
* Severity Comparisons

---

## Evidence Dossiers

Generate explainable reports for flagged tickets.

Each dossier contains:

* Ticket summary
* Evidence signals
* Severity rationale
* Auditor-friendly explanation

---

# Project Structure

```text
.
├── data/
│   └── customer_support_tickets.csv
│
├── models/
│   └── sia_model/
│       ├── best/
│       │   ├── adapter_config.json
│       │   ├── adapter_model.safetensors
│       │   ├── tokenizer.json
│       │   └── tokenizer_config.json
│       │
│       ├── fusion_model.pkl
│       ├── metrics.json
│       ├── threshold.npy
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── outputs/
│   ├── ablation_table.json
│   ├── cluster_ids.npy
│   ├── emb_reduced.npy
│   ├── evidence_dossiers.json
│   ├── labeled_tickets.csv
│   └── predictions.csv
│
├── app.py
├── train_pipeline.py
├── predict.py
├── requirements.txt
├── notebook.ipynb
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/disagr413/Support_Integrity_Auditor.git
cd Support_Integrity_Auditor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch the Streamlit App

```bash
streamlit run app.py
```
```

Outputs:

```text
labeled_tickets.csv
fusion_model.pkl
threshold.npy
metrics.json
trained LoRA model
```

---

# Inference

Run prediction pipeline:

```bash
python predict.py
```

Outputs:

```text
predictions.csv
evidence_dossiers.json
```

---

# Sample Output

```json
{
  "ticket_id": "10452",
  "assigned_priority": "Low",
  "inferred_severity": "Critical",
  "mismatch_type": "Hidden Crisis",
  "confidence": 0.94
}
```

---

## Model Performance

The final DeBERTa-v3 + LoRA model was evaluated on a held-out test set.

| Metric | Score |
|----------|----------|
| Accuracy | 85.4% |
| Macro F1 Score | 0.84 |
| Recall (Consistent) | 0.88 |
| Recall (Mismatch) | 0.81 |
| Optimized Threshold | 0.47 |

### Success Criteria

| Requirement | Target | Achieved |
|------------|---------|---------|
| Accuracy | ≥ 83% | ✅ 85.4% |
| Macro F1 | ≥ 0.82 | ✅ 0.84 |
| Recall (Consistent) | ≥ 0.78 | ✅ 0.88 |
| Recall (Mismatch) | ≥ 0.78 | ✅ 0.81 |

The model exceeds all predefined evaluation requirements.

---

## Key Capabilities Comparison

| Capability | Rule-Based | Traditional ML | SIA |
|------------|------------|------------|------------|
| Keyword Detection | ✅ | ✅ | ✅ |
| Semantic Understanding | ❌ | Partial | ✅ |
| Explainable Decisions | ✅ | Partial | ✅ |
| Evidence Generation | ❌ | ❌ | ✅ |
| Hidden Crisis Detection | Limited | Moderate | Strong |
| False Alarm Detection | Limited | Moderate | Strong |

SIA combines multi-signal pseudo-label generation, semantic ticket understanding, and transformer-based classification to achieve superior performance while maintaining explainability.

---

# Key Innovations

### Multi-Signal Pseudo Labeling

Combines independent severity signals instead of relying on manual labels.

### Semantic Ticket Understanding

Captures contextual severity beyond keyword matching.

### Parameter-Efficient Fine-Tuning

Uses LoRA for scalable deployment.

### Explainable AI

Generates evidence-backed audit reports.

### Operational Audit Focus

Designed specifically for support quality assurance and prioritization governance.

---

# Technology Stack

### Machine Learning

* PyTorch
* Transformers
* PEFT (LoRA)
* Scikit-Learn

### NLP

* DeBERTa-v3
* Sentence Transformers
* TF-IDF + SVD fallback

### Data Processing

* Pandas
* NumPy

### Clustering

* HDBSCAN
* UMAP
* K-Means

### Frontend

* Streamlit

---

# Future Improvements

* Active Learning Loop
* Human-in-the-Loop Validation
* Multi-Language Support
* Real-Time Ticket Monitoring
* Enterprise Ticketing Integrations
* LLM-Based Root Cause Analysis

---

# Authors

DISHA AGRAWAL
24113039
