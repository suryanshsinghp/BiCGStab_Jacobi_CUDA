// g++ -std=c++17 -o out bicgstab.cpp && ./out > cpu_20000.log
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstring> //for memcpy
#include <fstream>
#include <chrono>

// #define DEBUG_PRINT
// #define PRINT_TRUE_RESIDUAL

#define ARR_SIZE 20000
#define ARR_SIZE2 ARR_SIZE *ARR_SIZE

double dotProduct(double *a, double *b, int size)
{
    double sum = 0.0;
    for (int i = 0; i < size; ++i)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

double l2norm(double *input, int size)
{
    return sqrt(dotProduct(input, input, size));
}

void MatrixVecMult(double *A, double *x, double *result, int size)
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

void APAlB(double *a, double *b, double alpha, int size, double *result)
{ // res=A+alpha*B
    for (int i = 0; i < size; i++)
    {
        result[i] = a[i] + alpha * b[i];
    }
}

void vecSubtract(double *a, double *b, int size, double *result)
{
    for (int i = 0; i < size; ++i)
    {
        result[i] = a[i] - b[i];
    }
}

void JacobiPrecondition(double *input, double *M, double *result, int size)
{
    for (int i = 0; i < size; ++i)
    {
        result[i] = input[i] / M[i * size + i]; // M is preconditioner and diagonal of A
    }
}

double BiCGStab_with_Jacobi(double *A, double *b, double *x, int size, int maxIter, double tol)
{
    double *r = new double[size];
    double *r_hat = new double[size];
    double *p = new double[size];
    double *v = new double[size];
    double *s = new double[size];
    double *t = new double[size];
    double *p_precond = new double[size];
    double *s_precond = new double[size];
    double residual_norm;
    double r_res_norm;

    MatrixVecMult(A, x, r, size); // r = Ax
#ifdef DEBUG_PRINT
    std::cout << "initial r: " << r[1] << std::endl;
#endif
    vecSubtract(b, r, size, r); // r = b - Ax

    memcpy(r_hat, r, size * sizeof(double)); // r_hat = r
    memcpy(p, r, size * sizeof(double));     // p = r
    memset(v, 0, size * sizeof(double));     // v = 0
    double rho_old = 1.0, alpha = 1.0, omega = 1.0;

    for (int iter = 0; iter < maxIter; iter++)
    {
#ifdef DEBUG_PRINT
        std::cout << "Iteration: " << iter << std::endl;
#endif
        double rho = dotProduct(r_hat, r, size);
        if (fabs(rho) < 1e-20)
        {
#ifdef DEBUG_PRINT
            std::cout << "rho: " << rho << std::endl;
#endif
            break;
        }

        if (iter > 0)
        {
            double beta = (rho / rho_old) * (alpha / omega);
            APAlB(p, v, -omega, size, p);
            APAlB(r, p, beta, size, p); // p = r + beta * (p - omega * v)
            // AlXpBeYpc(beta, p, -omega*beta, v,r,p, size); // p = r + beta * (p - omega * v)
        }

        JacobiPrecondition(p, A, p_precond, size);
        MatrixVecMult(A, p_precond, v, size); // v = A * p_precond

        alpha = rho / dotProduct(r_hat, v, size);
        APAlB(r, v, -alpha, size, s); // s = r - alpha * v

        if (l2norm(s, size) < tol)
        {
            APAlB(x, p_precond, alpha, size, x); // x += alpha * p_precond
            residual_norm = l2norm(s, size);
#ifdef DEBUG_PRINT
            std::cout << "s array: " << l2norm(s, size) << std::endl;
#endif
            break;
        }

        JacobiPrecondition(s, A, s_precond, size);
        MatrixVecMult(A, s_precond, t, size); // t = A * s_precond

        double omega_num = dotProduct(t, s, size);
        double omega_den = dotProduct(t, t, size);
        if (fabs(omega_den) < 1e-20)
        {
#ifdef DEBUG_PRINT
            std::cout << "omega_den: " << omega_den << std::endl;
#endif
            break;
        }

        omega = omega_num / omega_den;

        APAlB(x, p_precond, alpha, size, x); // x += alpha * p_precond + omega * s_precond
        APAlB(x, s_precond, omega, size, x);
        // AlXpBeYpc(alpha, p_precond, omega, s_precond, x, x, size); // x += alpha * p_precond + omega * s_precond

        APAlB(s, t, -omega, size, r); // r = s - omega * t
        r_res_norm = l2norm(r, size);
        if (r_res_norm < tol)
        {
            residual_norm = r_res_norm;
#ifdef DEBUG_PRINT
            std::cout << "residual at convergence: " << l2norm(r, size) << std::endl;
#endif
            break;
        }

        rho_old = rho;
        std::cout << "iteration: " << iter << "    and    r residual :" << r_res_norm << std::endl;
    }

    delete[] r;
    delete[] r_hat;
    delete[] p;
    delete[] v;
    delete[] s;
    delete[] t;
    delete[] p_precond;
    delete[] s_precond;

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

    int maxIter = 5000;
    double tol = 1e-5;
    auto start = std::chrono::high_resolution_clock::now();
    double residual_norm = BiCGStab_with_Jacobi(A, b, x, ARR_SIZE, maxIter, tol);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "BiCGStab execution time: " << duration.count() / 1000.0 << " seconds." << std::endl;

    if (residual_norm > tol)
    {
        std::cerr << "BiCGStab did not converge!" << std::endl;
        return 1;
    }
    std::cout << "Converged with residual norm: " << residual_norm << std::endl;
    // write soultion
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
    MatrixVecMult(A, x, error_vec, ARR_SIZE);
    vecSubtract(error_vec, b, ARR_SIZE, error_vec);
    double error = l2norm(error_vec, ARR_SIZE);
    std::cout << "True Error Norm (Ax-b): " << error << std::endl;
#endif

    delete[] A;
    delete[] b;
    delete[] x;

    auto global_end = std::chrono::high_resolution_clock::now();
    auto global_duration = std::chrono::duration_cast<std::chrono::milliseconds>(global_end - global_start);
    std::cout << "Total execution time: " << global_duration.count() / 1000.0 << " seconds." << std::endl;
    return 0;
}