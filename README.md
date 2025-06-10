# BiCGStab with Jacobi pre-conditioner

Bi-conjugate gradient stabilized (BiCGStab) algorithm is implemented with Jacobi as a pre-conditioner in C++ and CUDA to solve for x in $Ax=b$ system of equations. This algorithm projects the problem in Krylov subspace and searches for an solution in this subspace to minimize the residual. The matrix “A” need not be symmetric and the full matrix is stored rather than using csr format since it is not necessary for the matrix to be sparse. However, depending on the problem, csr format can easily be integrated.  Further optimization will be made such as substituting atomics with faster reduction algorithms and using shared memory to speed up calculations (or cuBLAS libraries). The current speedup between the CPU (`bicgstab.cpp`) and the GPU (`bicgstab_gpu.cu`) version is:

| A: 5000 X 5000 | time [s] | total iteration | time/iteration | Scale|
| -------------------------- | -------- | --------------- | -------------- | -----|
| CPU        |          |                 |                |      |
| GPU (CUDA) |          |                 |                | s    |
| GPU (cuBLAS) |          |                 |                | s    |

