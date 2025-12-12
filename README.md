# bbFMM1D-Python

A clean Python implementation of the **Black-Box Fast Multipole Method (BBFMM)** in one dimension using Chebyshev interpolation for fast kernel summation.

## Overview

This implementation provides an efficient O(N) or O(N log N) algorithm for computing kernel summations that would normally require O(N²) operations. The method uses Chebyshev interpolation and hierarchical tree structures to approximate interactions between sources and field points with arbitrary smooth kernels.

### Key Features

- **Arbitrary Kernel Support**: Works with any smooth kernel function through black-box evaluation
- **Hierarchical Tree Construction**: Efficient binary tree structure for 1D spatial decomposition  
- **Chebyshev Interpolation**: High-order accurate approximations using Chebyshev nodes
- **SVD Compression**: Low-rank approximations for multipole-to-local (M2L) translations
- **Efficient Complexity**: Achieves near-linear computational cost for large-scale problems

## Algorithm Background

The Black-Box Fast Multipole Method (BBFMM) is based on the seminal work:

> **Fong, W., & Darve, E. (2009)**. "The Black-Box Fast Multipole Method."  
> *Journal of Computational Physics*, 228(23), 8712-8725.

The method accelerates kernel summations of the form:

```
u(x_i) = Σ K(x_i, y_j) * q_j
```

where `K` is a smooth kernel function, `y_j` are source points with charges `q_j`, and `x_i` are field (target) points.

### Algorithm Components

1. **Tree Construction**: Hierarchical subdivision of the computational domain
2. **Upward Pass**: Compute multipole expansions (S2M, M2M operators)
3. **Translation**: Transfer information between well-separated clusters (M2L operators with SVD compression)
4. **Downward Pass**: Compute local expansions (L2L, L2T operators)
5. **Near-field**: Direct evaluation for nearby interactions

## Repository Structure

```
bbFMM1D-Python/
├── FMM_Main.py                    # Main FMM driver and orchestration
├── H2_1D_Tree.py                  # Tree data structure implementation
├── H2_1D_Node.py                  # Tree node class definition
├── H2_1D_Tree_Functions.py        # Core FMM operators (S2M, M2M, M2L, L2L, L2T)
├── kernel_Base.py                 # Base kernel class and interface
├── CustomKernels.py               # Example kernel implementations
├── Input/                         # Input data files
└── README.md                      # This file
```

## Installation

### Prerequisites

```bash
Python 3.7+
numpy
scipy
```

### Setup

```bash
git clone https://github.com/Vaishnavi-Raghupathi/bbFMM1D-Python.git
cd bbFMM1D-Python
pip install numpy scipy
```

## Usage

### Basic Example

```python
import numpy as np
from FMM_Main import run_fmm
from CustomKernels import LogKernel

# Generate random source and field points
N_source = 1000
N_field = 1000
sources = np.random.uniform(0, 1, N_source)
field_points = np.random.uniform(0, 1, N_field)
charges = np.random.randn(N_source)

# Define kernel and run FMM
kernel = LogKernel()
result = run_fmm(sources, field_points, charges, kernel, 
                 n_levels=5, n_chebyshev=6)

print(f"Computed {N_field} potentials from {N_source} sources")
```

### Custom Kernel Implementation

You can define custom kernels by inheriting from the base kernel class:

```python
from kernel_Base import KernelBase

class MyKernel(KernelBase):
    def evaluate(self, x, y):
        """
        Evaluate kernel K(x, y)
        
        Args:
            x: field point(s)
            y: source point(s)
        Returns:
            kernel evaluation K(x, y)
        """
        r = np.abs(x - y)
        return 1.0 / (r + 1e-10)  # Example: Coulomb-like kernel
```

## Parameters

### Tree Construction
- `n_levels`: Number of tree levels (controls resolution)
- `min_points`: Minimum points per leaf node
- `domain`: Computational domain [x_min, x_max]

### Interpolation
- `n_chebyshev`: Number of Chebyshev nodes per box (typically 4-8)
- `epsilon`: Target accuracy for SVD compression (default: 1e-6)

### Kernel Options
Built-in kernels include:
- Logarithmic kernel: `K(r) = log(r)`
- Inverse distance: `K(r) = 1/r`  
- Gaussian: `K(r) = exp(-r²)`
- Custom kernels via `kernel_Base.py`

## Performance

The implementation demonstrates:
- **Speedup**: 10-100x faster than direct O(N²) evaluation for N > 1000
- **Accuracy**: Relative errors typically < 1e-6 with appropriate parameters
- **Scalability**: Near-linear scaling with problem size

### Benchmark Example

```python
# Compare FMM vs Direct evaluation
N = 5000
sources = np.random.uniform(0, 1, N)
charges = np.random.randn(N)

# FMM evaluation
t_start = time.time()
result_fmm = run_fmm(sources, sources, charges, kernel)
t_fmm = time.time() - t_start

# Direct evaluation  
t_start = time.time()
result_direct = direct_evaluation(sources, sources, charges, kernel)
t_direct = time.time() - t_start

error = np.linalg.norm(result_fmm - result_direct) / np.linalg.norm(result_direct)
print(f"Speedup: {t_direct/t_fmm:.2f}x")
print(f"Relative error: {error:.2e}")
```

## Mathematical Details

### Chebyshev Interpolation

For a box `[a, b]`, the Chebyshev nodes are:

```
x_k = (a + b)/2 + (b - a)/2 * cos((2k + 1)π / (2n))
```

Interpolation provides exponential convergence for smooth kernels.

### Operator Definitions

- **S2M (Source-to-Multipole)**: Projects source charges onto multipole expansion
- **M2M (Multipole-to-Multipole)**: Transfers multipole from child to parent
- **M2L (Multipole-to-Local)**: Translates multipole to local expansion (far-field)
- **L2L (Local-to-Local)**: Transfers local expansion from parent to child  
- **L2T (Local-to-Target)**: Evaluates local expansion at field points

## Limitations and Future Work

### Current Limitations
- 1D implementation only (extension to 2D/3D planned)
- Single-threaded execution
- Limited to smooth kernels (oscillatory kernels require special treatment)

### Planned Enhancements
- [ ] Multi-dimensional support (2D/3D)
- [ ] Parallel/GPU acceleration
- [ ] Adaptive tree refinement
- [ ] Additional kernel libraries
- [ ] Visualization tools

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

Areas for contribution:
- Performance optimization
- Additional kernel implementations  
- Documentation improvements
- Test coverage expansion
- Multi-dimensional extensions

## Citation

If you use this code in your research, please cite:

```bibtex
@article{fong2009blackbox,
  title={The black-box fast multipole method},
  author={Fong, William and Darve, Eric},
  journal={Journal of Computational Physics},
  volume={228},
  number={23},
  pages={8712--8725},
  year={2009}
}
```

## License

This project is available for research and educational purposes.

## Acknowledgments

This implementation is inspired by the foundational work of Fong & Darve on the Black-Box Fast Multipole Method. The code serves as both a learning resource and a foundation for research into fast algorithms for kernel summations.

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub or contact the repository maintainer.

---

**Note**: This is a research-oriented implementation designed for clarity and educational purposes. For production use cases requiring maximum performance, consider specialized FMM libraries like BBFMM3D, ExaFMM, or pvfmm.