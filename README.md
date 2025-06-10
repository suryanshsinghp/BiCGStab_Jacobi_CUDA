# BiCGStab with Jacobi pre-conditioner

Bi-conjugate gradient stabilized (BiCGStab) algorithm is implemented with Jacobi as a pre-conditioner in C++ and CUDA to solve for x in $Ax=b$ system of equations. This algorithm projects the problem in Krylov subspace and searches for an solution in this subspace to minimize the residual. The matrix “A” need not be symmetric and the full matrix is stored rather than using csr format since it is not necessary for the matrix to be sparse. However, depending on the problem, csr format can easily be integrated. The time comparison between the CPU (`bicgstab.cpp`), the GPU-CUDA (`bicgstab_gpu.cu`) and GPU-cuBLAS (`bicgstab_gpu_cublas.cu`) version is (convergence and true error (average of $|Ax-b|$) stored in `*.log` files.):

| A: 5000 X 5000 | solver time [s] | total iteration | time/iteration | Scale|
| -------------------------- | -------- | --------------- | -------------- | -----|
| CPU        | 27.02 | 333 | 0.0811 | 45.81 |
| GPU (CUDA) | 0.865 | 343 | 0.0025 | 1.42 |
| GPU (cuBLAS) | 0.588 | 332 | 0.0018 | 1   |



| A: 20000 X 20000 | solver time [s] | total iteration | time/iteration | Scale |
| ---------------- | --------------- | --------------- | -------------- | ----- |
| CPU              | 2499            | 1995            | 1.2526         | 84.39 |
| GPU (CUDA)       | 85.52           | 2068            | 0.0414         | 2.78  |
| GPU (cuBLAS)     | 27.95           | 1883            | 0.0148         | 1     |
