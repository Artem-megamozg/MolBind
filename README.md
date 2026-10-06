# MolBind

**MolBind** — исследовательский проект по предсказанию взаимодействия малых молекул и белков с использованием методов глубокого обучения.

Проект посвящён задаче **drug discovery**: по структуре потенциального лекарственного соединения и аминокислотной последовательности белка модель оценивает силу их связывания.

Основная задача первой версии — предсказание **pKd** на основе экспериментальных данных BindingDB.

---

## Research Objective

Цель проекта — исследовать, насколько эффективно совместное представление молекул и белков позволяет предсказывать binding affinity и обобщаться на ранее невиданные лекарственные соединения.

В проекте рассматриваются:

* графовые представления молекул;
* Transformer-based представление белковых последовательностей;
* совместное embedding space для ligand и protein;
* regression по pKd;
* ligand-cold evaluation;
* последующее сравнение различных архитектур и protein representations.

---

## Dataset

Основной источник данных:

**BindingDB**

Из исходного набора формируется датасет взаимодействий с экспериментально измеренным `Kd`.

Для первой версии используются только записи, удовлетворяющие условиям качества:

* валидный SMILES;
* валидная аминокислотная последовательность;
* известное значение Kd;
* single-chain target;
* допустимый диапазон pKd.

Целевая переменная:

```text
pKd = -log10(Kd [M])
```

### Current dataset

После preprocessing:

| Property       |  Value |
| -------------- | -----: |
| Interactions   | 56,130 |
| Unique ligands | 28,935 |
| Unique targets |  3,794 |

Dataset разделён на:

```text
Train       39,581
Validation   8,807
Test         7,742
```

Для текущего эксперимента используется **ligand-cold split**.

Пересечение молекул между наборами:

```text
Train ↔ Validation = 0
Train ↔ Test       = 0
Validation ↔ Test  = 0
```

При этом одинаковые protein targets могут присутствовать в разных наборах — это соответствует постановке ligand-cold evaluation.

---

## Model Architecture

Текущая архитектура состоит из двух независимых encoder'ов.

```text
                 ┌──────────────────────┐
SMILES ─────────►│ Molecular Graph      │
                 │ GATv2 Encoder        │
                 └──────────┬───────────┘
                            │
                            ▼
                       Molecule
                       Embedding
                            │
                            │
                       Fusion Network
                            │
                            ▼
                          pKd
                            ▲
                            │
                       Protein
                       Embedding
                            │
                 ┌──────────┴───────────┐
Protein Sequence►│ Transformer Encoder   │
                 │ + Positional Encoding │
                 └───────────────────────┘
```

### Molecular encoder

Молекула преобразуется из SMILES в граф:

```text
Atoms → Nodes
Bonds → Edges
```

Для построения node features используются:

* atomic number;
* atom degree;
* formal charge;
* aromaticity;
* hybridization;
* number of hydrogens.

Для bonds используются:

* bond type;
* conjugation;
* ring membership.

После этого граф проходит через несколько `GATv2Conv` layers.

### Protein encoder

Аминокислотная последовательность преобразуется в токены и обрабатывается Transformer Encoder.

Используются:

* amino-acid embeddings;
* positional embeddings;
* multi-head self-attention;
* masked mean pooling.

Максимальная длина последовательности в текущей конфигурации:

```text
512 amino acids
```

---

## Fusion

После получения embeddings формируются четыре компонента:

```text
M = molecule embedding
P = protein embedding

M
P
M × P
|M - P|
```

Они объединяются и передаются в MLP regression head.

Итог:

```text
Molecule + Protein → Interaction representation → pKd
```

---

## Training

Для обучения используется:

* PyTorch;
* PyTorch Geometric;
* AdamW;
* MSE loss;
* ReduceLROnPlateau;
* gradient clipping;
* early stopping;
* mixed precision при наличии CUDA.

Основные evaluation metrics:

```text
RMSE
MAE
R²
Pearson correlation
Spearman correlation
```

---

## Current Experiment

Первый запуск модели показал на validation set:

| Epoch |   RMSE |    MAE |     R² | Pearson | Spearman |
| ----: | -----: | -----: | -----: | ------: | -------: |
|     1 | 1.2724 | 0.9874 | 0.2999 |  0.5642 |   0.5447 |
|     2 | 1.2937 | 0.9840 | 0.2763 |  0.6058 |   0.5847 |
|     3 | 1.1926 | 0.9250 | 0.3850 |  0.6245 |   0.5985 |

Это **промежуточные результаты**, а не финальный benchmark.

Первый запуск выполнялся на CPU, поэтому experiment был остановлен для переноса обучения на CUDA GPU.

---

## Project Structure

```text
MolBind/
├── configs/
│   └── base.yaml
│
├── data/
│   ├── raw/
│   │   └── BindingDB_All.tsv
│   └── processed/
│       ├── bindingdb_pkd.parquet
│       ├── train.parquet
│       ├── validation.parquet
│       ├── test.parquet
│       └── metadata.json
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   │   ├── pkd_distribution.png
│   │   ├── protein_length_distribution.png
│   │   ├── samples_per_ligand.png
│   │   ├── samples_per_target.png
│   │   ├── top_targets.png
│   │   └── pkd_vs_protein_length.png
│   ├── best_model.pt
│   ├── metrics.json
│   └── training_history.json
│
├── scripts/
│   ├── prepare_bindingdb.py
│   ├── eda_bindingdb.py
│   ├── test_pipeline.py
│   └── train.py
│
├── src/
│   └── molbind/
│       ├── data/
│       │   ├── dataset.py
│       │   ├── preprocessing.py
│       │   └── seed.py
│       │
│       ├── features/
│       │   ├── molecule.py
│       │   └── protein.py
│       │
│       ├── models/
│       │   ├── encoders.py
│       │   └── model.py
│       │
│       ├── evaluation/
│       │   └── metrics.py
│       │
│       └── visualization/
│
├── tests/
├── pyproject.toml
└── README.md
```

---

## Data Pipeline

```text
BindingDB
   ↓
Raw TSV
   ↓
Cleaning
   ↓
SMILES validation
   ↓
Protein validation
   ↓
Kd filtering
   ↓
pKd transformation
   ↓
Duplicate aggregation
   ↓
Ligand-cold split
   ↓
Parquet datasets
   ↓
Graph + sequence encoding
   ↓
MolBind
```

---

## Experiments Roadmap

План дальнейшего исследования:

### Baselines

* molecular descriptors + protein descriptors;
* fingerprint-based baseline;
* simple MLP;
* GNN without protein Transformer.

### Architecture Experiments

* GAT vs GCN;
* GATv2 depth;
* different fusion strategies;
* cross-attention between molecule and protein;
* deeper protein Transformer.

### Protein Representations

Отдельно будет исследована замена собственного protein encoder на pretrained protein language models, например ESM-2.

```text
Sequence Transformer
        vs
ESM-2 embeddings
```

### Generalization

Будут исследованы разные стратегии разделения данных:

```text
Ligand-cold
Target-cold
Scaffold split
```

Это позволит оценить не только fit на существующие взаимодействия, но и способность модели работать с новыми химическими структурами и мишенями.

---

## Visualization

Проект предусматривает отдельный слой визуального анализа.

Планируемые визуализации:

* distribution of pKd;
* protein length distribution;
* ligand/target frequency;
* predicted vs actual affinity;
* residual analysis;
* training dynamics;
* embedding projections;
* error analysis;
* comparison of model variants.

---

## Research Questions

Основные вопросы проекта:

1. Насколько хорошо графовое представление молекулы совместно с protein sequence позволяет предсказывать binding affinity?
2. Насколько сильно pretrained protein embeddings улучшают качество?
3. Как меняется качество при переходе от random-like evaluation к ligand-cold, target-cold и scaffold splits?
4. Какие архитектуры fusion лучше моделируют molecule-protein interaction?
5. Можно ли использовать модель для ранжирования потенциальных кандидатов по predicted affinity?

---

## Tech Stack

```text
Python
PyTorch
PyTorch Geometric
RDKit
Pandas
NumPy
Scikit-learn
SciPy
Matplotlib
PyYAML
```

---

## Status

**Current stage:**

```text
[x] Project architecture
[x] BindingDB preprocessing
[x] EDA
[x] Ligand-cold split
[x] Molecular graph pipeline
[x] Protein tokenizer
[x] PyTorch Dataset
[x] DataLoader
[x] GATv2 encoder
[x] Protein Transformer
[x] Fusion model
[x] Training pipeline
[x] Initial validation experiment
[ ] CUDA training
[ ] Final benchmark
[ ] Baseline comparison
[ ] ESM-2 experiment
[ ] Target-cold evaluation
[ ] Scaffold split
[ ] Ablation study
[ ] Candidate ranking
```

---

## Disclaimer

MolBind is a **research project** intended for experimentation with machine learning methods in computational drug discovery.

Predicted binding affinity is not equivalent to experimentally confirmed biological activity, therapeutic efficacy, selectivity, safety, or clinical potential.
