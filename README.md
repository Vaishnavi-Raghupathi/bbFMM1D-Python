# 1D Black-Box Fast Multipole Method (BBFMM)

This repository explores a research-level implementation of the **1D Black-Box Fast Multipole Method (BBFMM)** based on the formulation presented in:

**Fong & Darve (2009) — "The Black-Box Fast Multipole Method"**

The goal of this project is to:
- Implement Chebyshev interpolation operators (S₂M, M2M, M2L, L2L, L2T)
- Construct a hierarchical 1D binary tree for source/target clustering
- Generate compressed M2L translation operators using SVD
- Compare BBFMM performance against direct O(N²) matrix–vector multiplication
- Explore numerical stability challenges in 1D FMM settings

This code is part of a broader experiment toward building a **research agentic pipeline** capable of reading research papers, extracting formulas, and generating working code implementations.

---

## Features

- 1D binary tree construction  
- Chebyshev/GLL node generation  
- Interpolation matrix generation  
- Multipole and local expansion propagation  
- SVD–compressed M2L operators  
- Direct vs BBFMM accuracy benchmarking  

---

## Repository Structure
```
bbfmm1d.py       # Main BBFMM implementation
test_bbfmm.py    # Benchmark script
utilities.py     # Chebyshev utilities (if separated)
README.md        # Documentation
```

---

## Running
```bash
python test_bbfmm.py
```

**Outputs:**
- Direct computation time
- FMM computation time
- Relative L2 error
- Maximum absolute error

---

## Disclaimer

This implementation is experimental and may not yet achieve the full numerical stability or performance of the original BBFMM algorithm.

The purpose of this repository is research, exploration, and iterative refinement.