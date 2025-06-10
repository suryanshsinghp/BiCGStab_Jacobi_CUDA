// nvcc -o out bicgstab_gpu.cu -arch=sm_60 && ./out > gpu_cuda_20000.log
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstring> //for memcpy
#include <fstream>
#include <chrono>

// #define DEBUG_PRINT
#define PRINT_TRUE_RESIDUAL

#define ARR_SIZE 20000
#define ARR_SIZE2 ARR_SIZE *ARR_SIZE

void MatrixVecMult_cpu(double *A, double *x, double *result, int size)
{
    for (int i = 0; i < size; ++i)
    {
        result[i] = 0.0;
        for (int j = 0; j < size; ++j)
        {
            result[i] += A[i * size + j] * x[j]; // remember A is 2D matrix that is flattened
        }
    }
}

void VecAvg_cpu(double *vec, int size, double *result)
{
    double sum = 0.0;
    for (int i = 0; i < size; ++i)
    {
        sum += vec[i];
    }
    *result = sum / size;
}

__global__ void dotProduct(double *a, double *b, int size, double *sum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        atomicAdd(sum, a[idx] * b[idx]);
    }
}

__global__ void MatrixVecMult(double *A, double *x, double *result, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        double sum = 0.0;
        for (int j = 0; j < size; ++j)
        {
            sum += A[idx * size + j] * x[j]; // remember A is 2D matrix that is flattened
        }
        result[idx] = sum;
    }
}

__global__ void APAlB(double *a, double *b, double alpha, int size, double *result)
{ // res=A+alpha*B
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = a[idx] + alpha * b[idx];
    }
}

__global__ void vecSubtract(double *a, double *b, int size, double *result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = a[idx] - b[idx];
    }
}

__global__ void JacobiPrecondition(double *input, double *M, double *result, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = input[idx] / M[idx * size + idx]; // M is preconditioner and diagonal of A
    }
}

double BiCGStab_with_Jacobi(double *d_A, double *d_b, double *d_x, int size, int maxIter, double tol)
{
    double *d_r, *d_r_hat, *d_p, *d_v, *d_s, *d_t, *d_p_precond, *d_s_precond, *d_sum;
    cudaMalloc(&d_r, size * sizeof(double));
    cudaMalloc(&d_r_hat, size * sizeof(double));
    cudaMalloc(&d_p, size * sizeof(double));
    cudaMalloc(&d_v, size * sizeof(double));
    cudaMalloc(&d_s, size * sizeof(double));
    cudaMalloc(&d_t, size * sizeof(double));
    cudaMalloc(&d_p_precond, size * sizeof(double));
    cudaMalloc(&d_s_precond, size * sizeof(double));
    cudaMalloc(&d_sum, sizeof(double));

    double residual_norm = 0.0;
    double r_res_norm = 0.0;

    int numThreads = 256;
    int numBlocks = (size + numThreads - 1) / numThreads; // ceil (size / numThreads)

    MatrixVecMult<<<numBlocks, numThreads>>>(d_A, d_x, d_r, size); // r = Ax
#ifdef DEBUG_PRINT
    cudaDeviceSynchronize();
    {
        double *h_r = new double[size];
        cudaMemcpy(h_r, d_r, size * sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "updated r (Ax):" << std::endl;
        for (int i = 0; i < size; ++i)
        {
            std::cout << h_r[i] << " ";
        }
        std::cout << std::endl;
    }
#endif
    cudaDeviceSynchronize();
    vecSubtract<<<numBlocks, numThreads>>>(d_b, d_r, size, d_r); // r = b - Ax
#ifdef DEBUG_PRINT
    cudaDeviceSynchronize();
    {
        double *h_r = new double[size];
        cudaMemcpy(h_r, d_r, size * sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "updated r (b-Ax):" << std::endl;
        for (int i = 0; i < size; ++i)
        {
            std::cout << h_r[i] << " ";
        }
        std::cout << std::endl;
    }
#endif

    cudaDeviceSynchronize();

    cudaMemcpy(d_r_hat, d_r, size * sizeof(double), cudaMemcpyDeviceToDevice); // r_hat = r
    cudaMemcpy(d_p, d_r, size * sizeof(double), cudaMemcpyDeviceToDevice);     // p = r
    cudaMemset(d_v, 0, size * sizeof(double));                                 // v = 0

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;

    for (int iter = 0; iter < maxIter; iter++)
    {
        cudaMemset(d_sum, 0, sizeof(double));                             // reset sum for dot product
        dotProduct<<<numBlocks, numThreads>>>(d_r_hat, d_r, size, d_sum); // rho = r_hat . r
        cudaDeviceSynchronize();
        double rho;
        cudaMemcpy(&rho, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
        if (fabs(rho) < 1e-20)
        {
            std::cout << "rho before breaking: " << rho << std::endl;
            break;
        }

        if (iter > 0)
        {
            double beta = (rho / rho_old) * (alpha / omega);
            APAlB<<<numBlocks, numThreads>>>(d_p, d_v, -omega, size, d_p);
            cudaDeviceSynchronize();
            APAlB<<<numBlocks, numThreads>>>(d_r, d_p, beta, size, d_p); // p = r + beta * (p - omega * v)
            cudaDeviceSynchronize();
        }

        JacobiPrecondition<<<numBlocks, numThreads>>>(d_p, d_A, d_p_precond, size);
        cudaDeviceSynchronize();
        MatrixVecMult<<<numBlocks, numThreads>>>(d_A, d_p_precond, d_v, size); // v = A * p_precond
        cudaDeviceSynchronize();

        cudaMemset(d_sum, 0, sizeof(double));                             // reset sum for dot product
        dotProduct<<<numBlocks, numThreads>>>(d_r_hat, d_v, size, d_sum); // rho = r_hat . v
        cudaDeviceSynchronize();
        double temp_sum;
        cudaMemcpy(&temp_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

        alpha = rho / temp_sum;
        APAlB<<<numBlocks, numThreads>>>(d_r, d_v, -alpha, size, d_s); // s = r - alpha * v
        cudaDeviceSynchronize();

        cudaMemset(d_sum, 0, sizeof(double));
        dotProduct<<<numBlocks, numThreads>>>(d_s, d_s, size, d_sum);
        cudaDeviceSynchronize();
        double s_norm;
        cudaMemcpy(&s_norm, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

        if (sqrt(s_norm) < tol)
        {
            APAlB<<<numBlocks, numThreads>>>(d_x, d_p_precond, alpha, size, d_x); // x += alpha * p_precond
            cudaDeviceSynchronize();
            cudaMemset(d_sum, 0, sizeof(double));
            dotProduct<<<numBlocks, numThreads>>>(d_s, d_s, size, d_sum);
            cudaDeviceSynchronize();
            cudaMemcpy(&residual_norm, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
            residual_norm = sqrt(residual_norm);
            break;
        }

        JacobiPrecondition<<<numBlocks, numThreads>>>(d_s, d_A, d_s_precond, size);
        cudaDeviceSynchronize();
        MatrixVecMult<<<numBlocks, numThreads>>>(d_A, d_s_precond, d_t, size); // t = A * s_precond
        cudaDeviceSynchronize();

        cudaMemset(d_sum, 0, sizeof(double));
        dotProduct<<<numBlocks, numThreads>>>(d_t, d_s, size, d_sum);
        cudaDeviceSynchronize();
        double omega_num;
        cudaMemcpy(&omega_num, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemset(d_sum, 0, sizeof(double));
        dotProduct<<<numBlocks, numThreads>>>(d_t, d_t, size, d_sum);
        cudaDeviceSynchronize();
        double omega_den;
        cudaMemcpy(&omega_den, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

        if (fabs(omega_den) < 1e-20)
        {
            break;
        }

        omega = omega_num / omega_den;

        APAlB<<<numBlocks, numThreads>>>(d_x, d_p_precond, alpha, size, d_x); // x += alpha * p_precond + omega* s_precond
        cudaDeviceSynchronize();
        APAlB<<<numBlocks, numThreads>>>(d_x, d_s_precond, omega, size, d_x);
        cudaDeviceSynchronize();

        APAlB<<<numBlocks, numThreads>>>(d_s, d_t, -omega, size, d_r); // r = s - omega * t
        cudaDeviceSynchronize();

        cudaMemset(d_sum, 0, sizeof(double));
        dotProduct<<<numBlocks, numThreads>>>(d_r, d_r, size, d_sum);
        cudaDeviceSynchronize();
        cudaMemcpy(&r_res_norm, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
        r_res_norm = sqrt(r_res_norm); // compute the L2 norm of the residual

        if (r_res_norm < tol)
        {
            residual_norm = r_res_norm;
            break;
        }

        rho_old = rho;
        std::cout << "iteration: " << iter << "    and    r residual :" << r_res_norm << std::endl;
    }

    cudaFree(d_r);
    cudaFree(d_r_hat);
    cudaFree(d_p);
    cudaFree(d_v);
    cudaFree(d_s);
    cudaFree(d_t);
    cudaFree(d_p_precond);
    cudaFree(d_s_precond);
    cudaFree(d_sum);

    return residual_norm;
}

int main()
{
    auto global_start = std::chrono::high_resolution_clock::now();
    double *A = new double[ARR_SIZE2];
    double *b = new double[ARR_SIZE];
    double *x = new double[ARR_SIZE];
    for (int i = 0; i < ARR_SIZE; ++i)
    {
        x[i] = rand() / ((float)RAND_MAX); // Initialize x to random values or zeros
    }

#ifdef READ_MATRIX_FROM_FILE
    // read A from file which is in 2D format
    {
        std::ifstream infile("matrix_A.dat");
        if (!infile)
        {
            std::cerr << "Error reading matrix A" << std::endl;
            return 1;
        }
        for (int i = 0; i < ARR_SIZE; ++i)
        {
            for (int j = 0; j < ARR_SIZE; ++j)
            {
                infile >> A[i * ARR_SIZE + j];
            }
        }
        infile.close();
    } // scope operator so I dont have to define new ifstream

    {
        std::ifstream infile("matrix_b.dat"); // read b matrix
        if (!infile)
        {
            std::cerr << "Error reading matrix b" << std::endl;
            return 1;
        }
        for (int i = 0; i < ARR_SIZE; ++i)
        {
            infile >> b[i];
        }
        infile.close();
    }
#else
    for (int i = 0; i < ARR_SIZE; ++i)
    {
        for (int j = 0; j < ARR_SIZE; ++j)
        {
            if (i == j)
            {
                A[i * ARR_SIZE + j] = 10.0; // Diagonal elements
            }
            else if (i < j)
            {
                A[i * ARR_SIZE + j] = 1.0; // Upper triangular elements
            }
            else
            {
                A[i * ARR_SIZE + j] = 2.0; // Off-diagonal elements
            }
        }
    }
    for (int i = 0; i < ARR_SIZE; ++i)
    {
        b[i] = 3.0; // Vector b
    }
#endif

#ifdef DEBUG_PRINT
    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < ARR_SIZE; ++i)
    {
        for (int j = 0; j < ARR_SIZE; ++j)
        {
            std::cout << A[i * ARR_SIZE + j] << " ";
        }
        std::cout << std::endl;
    }
    // write A matrix to file
    {
        std::ofstream outfile("matrix_A_out.dat");
        for (int i = 0; i < ARR_SIZE; ++i)
        {
            for (int j = 0; j < ARR_SIZE; ++j)
            {
                outfile << A[i * ARR_SIZE + j] << " ";
            }
            outfile << std::endl;
        }
        outfile.close();
    }

    std::cout << "Vector b:" << std::endl;
    for (int i = 0; i < ARR_SIZE; ++i)
    {
        std::cout << b[i] << " ";
    }
    std::cout << std::endl;
#endif

    // device arrays allocation and copy
    double *d_A, *d_b, *d_x;
    cudaMalloc(&d_A, ARR_SIZE2 * sizeof(double));
    cudaMalloc(&d_b, ARR_SIZE * sizeof(double));
    cudaMalloc(&d_x, ARR_SIZE * sizeof(double));
    cudaMemcpy(d_A, A, ARR_SIZE2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, ARR_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, ARR_SIZE * sizeof(double), cudaMemcpyHostToDevice); /// already initialized to zero
    // allocation and copy done

    int maxIter = 5000;
    double tol = 1e-5;
    auto start = std::chrono::high_resolution_clock::now();
    double residual_norm = BiCGStab_with_Jacobi(d_A, d_b, d_x, ARR_SIZE, maxIter, tol);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "BiCGStab execution time: " << duration.count() / 1000.0 << " seconds." << std::endl;

    if (residual_norm > tol)
    {
        std::cerr << "BiCGStab did not converge!" << std::endl;
        return 1;
    }
    std::cout << "Converged with residual norm: " << residual_norm << std::endl;
    // get x back to host and write soultion to file
    cudaMemcpy(x, d_x, ARR_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    {
        std::ofstream outfile("solution_x.dat");
        if (!outfile)
        {
            std::cerr << "Error writing matrix x" << std::endl;
            return 1;
        }
        for (int i = 0; i < ARR_SIZE; ++i)
        {
            outfile << x[i] << std::endl;
        }
        outfile.close();
    }

#ifdef PRINT_TRUE_RESIDUAL
    double *error_vec = new double[ARR_SIZE];
    MatrixVecMult_cpu(A, x, error_vec, ARR_SIZE);
    double avg_error;
    for (int i = 0; i < ARR_SIZE; ++i)
    {
        error_vec[i] -= b[i]; // error = Ax - b
        // std::cout << error_vec[i] << "\n ";
    }
    VecAvg_cpu(error_vec, ARR_SIZE, &avg_error);
    std::cout << "Average error (Ax-b): " << avg_error << std::endl;
#endif

    delete[] A;
    delete[] b;
    delete[] x;
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);

    auto global_end = std::chrono::high_resolution_clock::now();
    auto global_duration = std::chrono::duration_cast<std::chrono::milliseconds>(global_end - global_start);
    std::cout << "Total execution time: " << global_duration.count() / 1000.0 << " seconds." << std::endl;
    return 0;
}