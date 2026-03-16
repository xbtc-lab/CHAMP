# CHAMP: A Coupled Hierarchical Atom-Motif Predictor

This repository provides the official PyTorch implementation for **CHAMP (Coupled Hierarchical Atom-Motif Predictor)**, a novel hierarchical Graph Neural Network framework designed to achieve state-of-the-art performance in molecular property prediction.

## Introduction

CHAMP is engineered to address two central challenges in GNN-based molecular science: achieving **Structural Completeness** in motif representations and ensuring **Functional Discriminability** through context-aware learning. It systematically overcomes the limitations of conventional models by introducing a dynamic **"guidance-fusion-regulation"** process that enables true **Synergistic Multi-scale Integration**.

By operating on a dual-view representation of molecules (atom-level and motif-level graphs), CHAMP learns to generate molecular representations that are not only structurally faithful but also highly sensitive to functional context, leading to superior predictive accuracy and enhanced chemical interpretability.

## Overall Architecture

The workflow of CHAMP unfolds in a three-stage hierarchical process designed to synergistically integrate fine-grained atomic details with coarse-grained functional semantics.

![overview.png](overview.png)

The framework is organized around three conceptual stages:

1. **Motif construction and structural encoding**
   CHAMP builds motif-level representations on top of atom-level molecular graphs and models internal motif topology to preserve structural information.

2. **Function-aware motif refinement**
   CHAMP refines motif embeddings through supervised contrastive constraints so that structurally similar motifs with different functional roles can be distinguished more effectively.

3. **Hierarchical atom-motif fusion**
   CHAMP uses motif-level semantics to guide atom-level aggregation and performs cross-scale fusion through gating and inter-head interaction mechanisms.

The current public release focuses on the core modules and the main training workflow implemented in this repository.

## Repository Scope

The released codebase includes:

- the core model components in `Model/`,
- motif extraction and motif-graph construction in `motif_extract/`,
- shared helper utilities in `utils/`,
- argument configuration in `Args.py`,
- motif-aware dataset preparation in `motif_spilit.py`,
- the main public training script in `main_classification.py`,
- the dependency specification in `requirements.txt`.

Local folders such as `dataset/`, `best_model/`, `.idea/`, and `__pycache__/` may appear in the working directory, but they should be interpreted as local resources or development artifacts rather than as the conceptual core of the released source implementation.

## Repository Structure

The current directory structure of the released code is:

```text
Code/
├── Args.py
├── main_classification.py
├── motif_spilit.py
├── overview.png
├── README.md
├── requirements.txt
├── Model/
│   ├── HMSAF.py
│   ├── atom_motif_attention.py
│   ├── contrastive_learning.py
│   └──  motif_embedding.py
└──  motif_extract/
    ├── mol_motif.py
    └──  motif_graph.py
```

For readers who only want to understand or reuse the main implementation, the primary source files are:

- `main_classification.py`
- `Args.py`
- `motif_spilit.py`
- `Model/*.py`
- `motif_extract/*.py`

## Environment

The released code was prepared for Python 3.9 and PyTorch-based execution.

Install dependencies with:

```bash
pip install -r requirements.txt
```
**Main dependencies:**

- PyTorch (1.12.0+cu113)
- PyTorch Geometric (2.6.1)
- RDKit (2024.9.3)
- scikit-learn (1.7.2)
- UMAP-learn (0.5.7)

## Usage

### Parameter Configuration

## Configuration

The argument configuration is defined in `Args.py`.

Important arguments in the current release include:

- `--dataset`: dataset name
- `--data_dir`: dataset directory
- `--node_feature_dim`: atom feature dimension
- `--edge_feature_dim`: edge feature dimension
- `--hidden_dim`: hidden representation dimension
- `--batch_size`: batch size
- `--epochs`: number of epochs
- `--lr`: learning rate
- `--weight_decay`: optimizer weight decay
- `--patience`: scheduler patience
- `--factor`: scheduler decay factor
- `--loss_fn`: loss function option
- `--alpha`: ring-level contrastive loss weight
- `--beta`: non-ring contrastive loss weight
- `--Pair_MLP`: whether to enable the pairwise motif encoder option
- `--is_contrastive`: whether to enable contrastive learning
- `--use_Guide`: whether to enable motif guidance
- `--use_gating`: whether to enable contextual gating
- `--use_head_interaction`: whether to enable inter-head interaction
- `--label_thresh_ratio`: threshold ratio used in motif comparison
- `--save_dir`: checkpoint directory
- `--log_dir`: log directory
- `--device`: execution device

### Supported Datasets

The framework supports a wide range of datasets from MoleculeNet, including:

- **Regression Tasks:** ESOL, FreeSolv, Lipophilicity.
- **Classification Tasks:** MUTAG, HIV, BACE, Tox21.

Datasets are expected in a standard graph format, containing node features, edge connectivity, and molecular labels.

### Running Experiments

```bash
python main_classification.py --dataset BACE --use_head_interaction True --use_gating True
