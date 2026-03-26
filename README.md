# T-MGS: Topology-Optimal Multiple Gossip Steps for Decentralized Federated Learning via Gossip Tensor

[![IEEE TNNLS](https://img.shields.io/badge/IEEE%20TNNLS-10.1109%2FTNNLS.2026.3670013-blue)](https://doi.org/10.1109/TNNLS.2026.3670013)
[![Python 3](https://img.shields.io/badge/Python-3-green)](https://www.python.org/)

This repository provides the official implementation code supporting the paper:

> **"Topology-Optimal Multiple Gossip Steps for Decentralized Federated Learning via Gossip Tensor"**  
> *IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2026*  
> DOI: [10.1109/TNNLS.2026.3670013](https://doi.org/10.1109/TNNLS.2026.3670013)

---

## Overview

In decentralized federated learning (DFL), nodes communicate with their neighbors via **gossip protocols**. Running multiple gossip steps per training round can significantly accelerate model consensus, but naively stacking gossip rounds leads to suboptimal mixing matrices.

This project implements **T-MGS (Topology-optimal Multiple Gossip Steps)**, a framework that:

- Constructs the **Gossip Tensor** `A` — a 3-dimensional tensor encoding the routing weights for each gossip step — such that the resulting multi-step mixing matrix is **doubly stochastic** and **topology-respecting**.
- Solves a **convex optimization** problem to find the gossip tensor `A_k` at each step `k`, minimizing the spectral gap of the composed mixing matrix.
- Supports a variety of **network topologies** commonly studied in decentralized learning literature.

### Key Concepts

| Symbol | Description |
|--------|-------------|
| `G` | Adjacency matrix of the communication graph |
| `A_k` | Gossip Tensor at step `k` (3D tensor, shape `[n, n, n]`) |
| `q_k` | Mixing matrix achieved after `k` gossip steps (doubly stochastic) |
| `Q_k` | Ideal (fastest-mixing) matrix on the `k`-hop graph |
| `K` | Total number of gossip steps |

---

## Supported Network Topologies

The following graph types are supported out of the box:

- **Erdős–Rényi (ER) Random Graph**
- **Ring Network**
- **Fixed-Degree Random Graph**
- **Grid Network**
- **Torus Network**
- **Complete Binary Tree**

---

## Installation

```bash
git clone https://github.com/YanZhong-Lab-ECNUstat/Tensor_Multiple_Gossip_Steps_T-MGS.git
cd T-MGS
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- `numpy` — array and matrix operations
- `cvxpy` — convex optimization for gossip tensor construction (SDP/SCS)
- `scipy` — sparse linear programming (HiGHS solver)
- `networkx` — graph generation and manipulation

---

## Quick Start

```python
from generate_graph import generate_ring
from tmgs import TMGSK

# Generate a ring graph with 30 nodes
G = generate_ring(30, 1)

# Solve for K=4 gossip steps
A_result, q_result, Q_result = TMGSK(G, K=4)

# A_result: list of gossip tensors [A_1, A_2, ..., A_{K-1}]
# q_result: list of achieved mixing matrices [q_2, q_3, ..., q_K]
# Q_result: list of ideal mixing matrices    [Q_2, Q_3, ..., Q_K]
```

Run the full benchmark across all supported topologies:

```bash
python main.py
```

---

## Module Description

### `generate_graph.py` — Graph Generation

Generates adjacency matrices for various network topologies used in decentralized learning experiments.

| Function | Description |
|----------|-------------|
| `generate_ER(n, pi)` | Erdős–Rényi random graph with `n` nodes and edge probability `pi` |
| `generate_ring(n, D)` | Ring graph where each node connects to `D` neighbors on each side |
| `generate_fixed_degree(n, D)` | Random `D`-regular graph |
| `generate_grid(n, cols)` | 2D grid graph with `n` nodes and `cols` columns |
| `generate_torus(n, cols)` | 2D torus (periodic grid) graph |
| `generate_complete_binary_tree(n)` | Complete binary tree with `n` nodes |

All adjacency matrices include **self-loops** (diagonal = 1) as required by the gossip protocol.

---

### `tmgs.py` — Core Algorithm

Implements the T-MGS algorithm for constructing the gossip tensor.

| Function | Description |
|----------|-------------|
| `getA_k(G, k)` | Computes the `k`-hop reachability matrix from adjacency matrix `G` |
| `fastest_mixing(G)` | Solves the **Fastest Mixing Markov Chain (FMMC)** problem on graph `G` via SDP to obtain the ideal doubly stochastic mixing matrix |
| `TMGS2(G)` | Constructs the gossip tensor `A_1` for `K=2` gossip steps using linear programming (HiGHS) |
| `mid_optimization(b, G, G_k)` | Iteratively constructs `A_k` for step `k ≥ 3` by solving a convex program (SCS solver) that minimizes the spectral norm of the composed mixing matrix |
| `TMGSK(G, K)` | **Main entry point.** Runs the full T-MGS pipeline for `K` gossip steps, returning all gossip tensors `A`, achieved mixing matrices `q`, and ideal mixing matrices `Q` |

**Algorithm flow of `TMGSK`:**
1. Solve `TMGS2` to get `A_1` and `q_2` (base case, `K=2`).
2. For each subsequent step `k = 3, ..., K`, call `mid_optimization` to solve for `A_{k-1}` and `q_k`.
3. Accumulate the composed path-weight tensor `b_k` via tensor contraction across gossip steps.

---

### `tmgs_check.py` — Correctness Verification

Provides validation utilities to verify that the computed gossip tensors and mixing matrices satisfy all theoretical constraints from the paper.

| Function | Description |
|----------|-------------|
| `check_A(A_list, G)` | Verifies that each gossip tensor `A_k` only assigns nonzero weights to edges present in `G` (topology constraint) |
| `check_q_k(q_list, G)` | Verifies that each mixing matrix `q_k` is **symmetric**, **non-negative**, **doubly stochastic**, and **supported on the `k`-hop graph** |
| `check_q_k_A_k(q_list, A_list, G)` | Verifies the core identity: the mixing matrix `q_k` equals the sum over all `k`-hop paths of the product of gossip tensor weights (Eq. 15 in the paper), validated via DFS path enumeration |

---

### `main.py` — Benchmark Entry Point

Runs T-MGS with `K=4` gossip steps across all six supported network topologies (30 nodes each) and prints constraint-satisfaction checks for each result.

---

## Citation

If you find this code useful in your research, please cite:

```bibtex
@ARTICLE{11429702,
  author={Zhong, Yan and Ma, Lei and Yan, Xiaomeng},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Topology-Optimal Multiple Gossip Steps for Decentralized Federated Learning via Gossip Tensor}, 
  year={2026},
  volume={},
  number={},
  pages={1-14},
  keywords={Convergence;Topology;Symmetric matrices;Eigenvalues and eigenfunctions;Tensors;Federated learning;Heuristic algorithms;Accuracy;Reviews;Privacy;Communication efficiency;communication round;decentralized graph;federated learning;second-largest absolute eigenvalue},
  doi={10.1109/TNNLS.2026.3670013}}
}
```

---

## Keywords

Convergence;Topology;Symmetric matrices;Eigenvalues and eigenfunctions;Tensors;Federated learning;Heuristic algorithms;Accuracy;Reviews;Privacy;Communication efficiency;communication round;decentralized graph;federated learning;second-largest absolute eigenvalue

