// nvcc -o out bicgstab_gpu_cublas.cu -lcublas && ./out > gpu_cublas_20000.log
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstring> //for memcpy
#include <fstream>
#include <chrono>
#include <cublas_v2.h>

// #define DEBUG_PRINT
#define PRINT_TRUE_RESIDUAL

#define ARR_SIZE 20000
#define ARR_SIZE2 ARR_SIZE *ARR_SIZE

// from NVIDIA, for error checking
#define cudaCheckErrors(msg)                                   \
    do                                                         \
    {                                                          \
        cudaError_t __err = cudaGetLastError();                \
        if (__err != cudaSuccess)                              \
        {                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err),            \
                    __FILE__, __LINE__);                       \
            fprintf(stderr, "*** FAILED - ABORTING\n");        \
            exit(1);                                           \
        }                                                      \
    } while (0)

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
    double *d_r, *d_r_hat, *d_p, *d_v, *d_s, *d_t, *d_p_precond, *d_s_precond;
    cudaMalloc(&d_r, size * sizeof(double));
    cudaMalloc(&d_r_hat, size * sizeof(double));
    cudaMalloc(&d_p, size * sizeof(double));
    cudaMalloc(&d_v, size * sizeof(double));
    cudaMalloc(&d_s, size * sizeof(double));
    cudaMalloc(&d_t, size * sizeof(double));
    cudaMalloc(&d_p_precond, size * sizeof(double));
    cudaMalloc(&d_s_precond, size * sizeof(double));
    cudaCheckErrors("Allocating 2");

    double one = 1.0;
    double one_neg = -1.0;
    double zero = 0.0;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t status;

    double residual_norm = 0.0;
    double r_res_norm = 0.0;

    int numThreads = 256;
    int numBlocks = (size + numThreads - 1) / numThreads; // ceil (size / numThreads)

    // MatrixVecMult<<<numBlocks, numThreads>>>(d_A, d_x, d_r, size); // r = Ax
    status = cublasDgemv(handle, CUBLAS_OP_T, size, size, &one_neg, d_A, size, d_x, 1, &zero, d_r, 1); // r = -Ax
#ifdef DEBUG_PRINT
    cudaDeviceSynchronize();
    {
        double *h_r = new double[size];
        cudaMemcpy(h_r, d_r, size * sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "updated r (-Ax):" << std::endl;
        for (int i = 0; i < size; ++i)
        {
            std::cout << h_r[i] << " ";
        }
        std::cout << std::endl;
    }
#endif
    // cudaDeviceSynchronize();
    // cublasDscal(handle, size, &one_neg, d_r, 1);     // r = - r
    status = cublasDaxpy(handle, size, &one, d_b, 1, d_r, 1); // r = b + r
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
        double rho;
#ifdef DEBUG_PRINT
        {
            double *h_r = new double[size];
            int num_elem_print = 10;
            cudaMemcpy(h_r, d_r, size * sizeof(double), cudaMemcpyDeviceToHost);
            std::cout << "First " <<num_elem_print<< " elements of r: ";
            for (int i = 0; i < num_elem_print && i < size; ++i)
                std::cout << h_r[i] << " ";
            std::cout << std::endl;
        }
#endif
        status = cublasDdot(handle, size, d_r_hat, 1, d_r, 1, &rho);
        // std::cout << "rho: " << rho << std::endl;
        cudaCheckErrors("dot product rho");
        if (fabs(rho) < 1e-20)
        {
            std::cout << "rho before breaking: " << rho << std::endl;
            break;
        }

        if (iter > 0)
        {
            double beta = (rho / rho_old) * (alpha / omega);
            double omega_neg = -1.0 * omega;
            cublasDaxpy(handle, size, &omega_neg, d_v, 1, d_p, 1); // p = p - omega * v
            cublasDscal(handle, size, &beta, d_p, 1); // p = beta * p
            cublasDaxpy(handle, size, &one, d_r, 1, d_p, 1); // p = r + p
        }
        cudaDeviceSynchronize();

        JacobiPrecondition<<<numBlocks, numThreads>>>(d_p, d_A, d_p_precond, size);
        cudaDeviceSynchronize();
        status = cublasDgemv(handle, CUBLAS_OP_T, size, size, &one, d_A, size, d_p_precond, 1, &zero, d_v, 1); // v = A * p_precond
        double temp_sum;
        status = cublasDdot(handle, size, d_r_hat, 1, d_v, 1, &temp_sum);

        alpha = rho / temp_sum;
        double alpha_neg = -1.0 * alpha;
        cudaMemcpy(d_s, d_r, size * sizeof(double), cudaMemcpyDeviceToDevice); // s = r
        status = cublasDaxpy(handle, size, &alpha_neg, d_v, 1, d_s, 1);        // s = s - alpha * v

        cudaDeviceSynchronize();

        double s_norm;
        status = cublasDdot(handle, size, d_s, 1, d_s, 1, &s_norm);

        if (sqrt(s_norm) < tol)
        {
            status = cublasDaxpy(handle, size, &alpha, d_p_precond, 1, d_x, 1); // x += alpha * p_precond
            status = cublasDdot(handle, size, d_s, 1, d_s, 1, &residual_norm);
            residual_norm = sqrt(residual_norm);
            break;
        }

        JacobiPrecondition<<<numBlocks, numThreads>>>(d_s, d_A, d_s_precond, size);
        cudaDeviceSynchronize();
        // MatrixVecMult<<<numBlocks, numThreads>>>(d_A, d_s_precond, d_t, size); // t = A * s_precond
        status = cublasDgemv(handle, CUBLAS_OP_T, size, size, &one, d_A, size, d_s_precond, 1, &zero, d_t, 1); // t = A * s_precond
        cudaDeviceSynchronize();

        double omega_num;
        status = cublasDdot(handle, size, d_t, 1, d_s, 1, &omega_num);
        double omega_den;
        status = cublasDdot(handle, size, d_t, 1, d_t, 1, &omega_den);

        if (fabs(omega_den) < 1e-20)
        {
            break;
        }

        omega = omega_num / omega_den;
        double omega_neg = -1.0 * omega;

        status = cublasDaxpy(handle, size, &alpha, d_p_precond, 1, d_x, 1); // x += alpha * p_precond
        status = cublasDaxpy(handle, size, &omega, d_s_precond, 1, d_x, 1); // x += omega * s_precond

        cudaMemcpy(d_r, d_s, size * sizeof(double), cudaMemcpyDeviceToDevice); // r = s
        status = cublasDaxpy(handle, size, &omega_neg, d_t, 1, d_r, 1);        // r = r - omega * t

        status = cublasDdot(handle, size, d_r, 1, d_r, 1, &r_res_norm);
        r_res_norm = sqrt(r_res_norm); // compute the L2 norm of the residual

        if (r_res_norm < tol)
        {
            residual_norm = r_res_norm;
            break;
        }

        rho_old = rho;
        std::cout << "iteration: " << iter << "    and    r residual :" << r_res_norm << std::endl;
        cudaCheckErrors("inside iteration loop");
    }

    cudaFree(d_r);
    cudaFree(d_r_hat);
    cudaFree(d_p);
    cudaFree(d_v);
    cudaFree(d_s);
    cudaFree(d_t);
    cudaFree(d_p_precond);
    cudaFree(d_s_precond);

    cublasDestroy(handle);

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

#ifdef PRINT_INPUT_MATRIX
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
    cudaCheckErrors("Allocating and copying to device 1");

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