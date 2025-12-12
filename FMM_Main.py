import numpy as np
from time import perf_counter
from CustomKernels import CustomKernels
from H2_1D_Tree import H2_1D_Tree
from kernel_Base import calculate_Potential

Data = np.loadtxt('Input/input.txt')
location = Data[:10000, 0:1]
charges = Data[:10000, 1:]
N = len(location)
m = charges.shape[1]
nChebNodes = 5

print(' Number of charges: %d' % N)
print(' Number of sets of charges: %d' % m)
print(' Number of Chebyshev Nodes: %d' % nChebNodes)

start = perf_counter()
ATree = H2_1D_Tree(nChebNodes, charges, location, N, m)
print(' Total time taken for FMM(build tree): %f seconds' % (perf_counter() - start))

start = perf_counter()
kernel = CustomKernels.laplacian1D
potential_fmm = calculate_Potential(kernel, ATree, charges)
print(' Total time taken for FMM(calculations): %f seconds' % (perf_counter() - start))

start = perf_counter()
print('\n Starting exact computation...')
Q = kernel(N, location, N, location)
potential_exact = np.dot(Q, charges)
print(' Done.')
print(' Total time taken for Exact(calculations): %f seconds' % (perf_counter() - start))

print('\n Maximum Error: %.3e\n' % (np.linalg.norm(potential_exact - potential_fmm) / np.linalg.norm(potential_exact)))