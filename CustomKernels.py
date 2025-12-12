import numpy as np
import math

class CustomKernels:
    
    @staticmethod
    def exampleKernelA(M, x, N, y):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        x0, y0 = np.meshgrid(y[:, 0], x[:, 0])
        kernel = np.abs(x0 - y0)
        return kernel
    
    @staticmethod
    def exampleKernelB(M, x, N, y):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        i, j = np.indices((M, N))
        kernel = np.empty((M, N))
        kernel[i, j] = np.abs(x[i, 0] - y[j, 0])
        return kernel
    
    @staticmethod
    def exampleKernelC(M, x, N, y):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        kernel = np.empty([M, N])
        for i in range(M):
            for j in range(N):
                kernel[i, j] = abs(x[i, 0] - y[j, 0])
        return kernel
    
    @staticmethod
    def laplacian1D(M, x, N, y):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        x0, y0 = np.meshgrid(y[:, 0], x[:, 0])
        r = np.abs(x0 - y0)
        with np.errstate(divide='ignore', invalid='ignore'):
            kernel = np.where(r > 1e-10, 1.0 / r, 0.0)
        return kernel
    
    @staticmethod
    def gaussian1D(M, x, N, y, a=1.0):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        x0, y0 = np.meshgrid(y[:, 0], x[:, 0])
        r = np.abs(x0 - y0)
        kernel = np.exp(-(r / a) ** 2)
        return kernel


if __name__ == '__main__':
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([[0.5], [1.5]])
    
    K = CustomKernels.laplacian1D(3, x, 2, y)
    print("Laplacian kernel test:")
    print(K)