// Лаб 3 - умножение матриц

#include <assert.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>

using namespace std;

//                CxR
// A принадлежит R
//
// Acr - элемент матрицы a в столбце c, строке r
// 
// c принадлежит Zc, r принадлежит Zr.
// 
//  C1 x R1    C2 x C1         C2 x R1 
// S        x S        -----> S
//
// ^          ^               ^
// |          |               |
// B          C               A
//
// 0 <= i <= C2 - 1
//                   => A aij = Сумма от l = 0 до l = C1 - 1 (b_lj * c_il)
// 0 <= j <= R1 - 1
//

// Matrices can be multiplied only if the length of the first 
    // matrix is equal to height of the second.
    //
    //                     <---k--->   <---k--->
    // <-------n------->   ^           ^
    // ^                   |           |
    // |                   |           m
    // m                 * n         = |
    // |                   |           v
    // v                   |
    //                     v

void mul_matrix(double* A, size_t cA, size_t rA,
    const double* B, size_t cB, size_t rB,
    const double* C, size_t cC, size_t rC)
{
    assert(cB == rC && cA == cC && rA == rB);
    assert((cA & 0x3f) == 0);

    for (size_t i = 0; i < cA; i++)
    {
        for (size_t j = 0; j < rA; j++)
        {
            A[i * rA + j] = 0;
            for (size_t k = 0; k < cB; k++)
            {
                A[i * rA + j] += B[k * rB + j] * C[i * rC + k];
            }
        }
    }
}

//_mm512_mul_pd
//_mm512_fmadd_pd
void mul_matrix_avx_512(double* A,
    size_t cA, size_t rA,
    const double* B,
    size_t cB, size_t rB,
    const double* C,
    size_t cC, size_t rC)
{
    assert(cB == rC && cA == cC && rA == rB);
    assert((cA & 0x3f) == 0);

    for (size_t i = 0; i < rB / 8; i++)
    {
        for (size_t j = 0; j < cC; j++)
        {
            __m512d sum = _mm512_setzero_pd();
            for (size_t k = 0; k < rC; k++)
            {
                __m512d bCol = _mm512_loadu_pd(B + rB * k + i * 8);
                __m512d broadcasted = _mm512_set1_pd(C[j * rC + k]);
                sum = _mm512_fmadd_pd(bCol, broadcasted, sum);
            }

            _mm512_storeu_pd(A + j * rA + i * 8, sum);
        }
    }
}

void mul_matrix_avx_256(double* A,
    size_t cA, size_t rA,
    const double* B,
    size_t cB, size_t rB,
    const double* C,
    size_t cC, size_t rC)
{
    assert(cB == rC && cA == cC && rA == rB);
    assert((cA & 0x3f) == 0);

    for (size_t i = 0; i < rB / 4; i++)
    {
        for (size_t j = 0; j < cC; j++)
        {
            __m256d sum = _mm256_setzero_pd();
            for (size_t k = 0; k < rC; k++)
            {
                __m256d bCol = _mm256_loadu_pd(B + rB * k + i * 4);
                __m256d broadcasted = _mm256_set1_pd(C[j * rC + k]);
                sum = _mm256_fmadd_pd(bCol, broadcasted, sum);
            }

            _mm256_storeu_pd(A + j * rA + i * 4, sum);
        }
    }
}

// Получение матрицы перестановки
pair<vector<double>, vector<double>> get_permutation_matrix(size_t n)
{
    vector<size_t> permut(n);

    for (size_t i = 0; i < n; i++)
    {
        permut[i] = (n - (i + 10)) % n;
    }

    vector<double> vf(n * n), vi(n * n);

    for (size_t c = 0; c < n; c++)
    {
        for (size_t r = 0; r < n; r++)
        {
            vf[c * n + r] = vi[r * n + c] = 1;
        }
    }

    return pair{ vf, vi }; // C++ 17
}

// Получение матрицы перестановки (другая реализация)
vector<double> generate_permutation_matrix(std::size_t n)
{
    vector<double> permut_matrix(n * n, 0);

    for (std::size_t i = 0; i < n; i++)
    {
        permut_matrix[(i + 1) * n - 1 - i] = 1;
    }

    return permut_matrix;
}

int main(int argc, char** argv)
{
    const std::size_t exp_count = 10;

    std::ofstream output("output.csv");
    if (!output.is_open())
    {
        std::cout << "Error!\n";
        return -1;
    }

    auto show_matrix = [](const double* A, std::size_t colsc, std::size_t rowsc)
    {
        for (std::size_t r = 0; r < rowsc; ++r)
        {
            cout << "[" << A[r + 0 * rowsc];
            for (std::size_t c = 1; c < colsc; ++c)
            {
                cout << ", " << A[r + c * rowsc];
            }
            cout << "]\n";
        }
        cout << "\n";
    };

    auto randomize_matrix = [](double* matrix, std::size_t matrix_order) {
        uniform_real_distribution<double> unif(0, 100000);
        default_random_engine re;
        for (size_t i = 0; i < matrix_order * matrix_order; i++)
        {
            matrix[i] = unif(re);
        }
    };

    const std::size_t matrix_order = 16 * 4 * 9;

    vector<double> A(matrix_order * matrix_order),
        C(matrix_order * matrix_order),
        D(matrix_order * matrix_order);
    vector<double> B = generate_permutation_matrix(matrix_order);
    // делали ещё так
    // auto [A, B] = get_permutation_matrix(matrix_order);

    //show_matrix(B.data(), matrix_order, matrix_order);
    randomize_matrix(A.data(), matrix_order);

    std::cout << "==Correctness test. ";
    //Perform naive multiplication.
    mul_matrix(C.data(), matrix_order, matrix_order,
        A.data(), matrix_order, matrix_order,
        B.data(), matrix_order, matrix_order);

    //Perform vectorized multiplication.
    mul_matrix_avx_256(D.data(), matrix_order, matrix_order,
        A.data(), matrix_order, matrix_order,
        B.data(), matrix_order, matrix_order);
    // mul_matrix_avx_512(D.data(), matrix_order, matrix_order,
       // A.data(), matrix_order, matrix_order,
       // B.data(), matrix_order, matrix_order);

    if (memcmp(static_cast<void*>(C.data()),
        static_cast<void*>(D.data()),
        matrix_order * matrix_order * sizeof(double)))
    {
        cout << "FAILURE==\n";
        output.close();
        return -1;
    }

    //Two possibilities: both functions work properly or both functions
    //fail in the same way. We pray that everything works.
    cout << "ok.==\n";
    //show_matrix(C.data(), matrix_order, matrix_order);
    //show_matrix(D.data(), matrix_order, matrix_order);

    std::cout << "==Performance tests.==\n";
    output << "Type,Duration,Acceleration\n";

    //Scalar multiplication. C = A*B.
    double duration = 0;
    for (std::size_t i = 0; i < exp_count; i++)
    {
        randomize_matrix(A.data(), matrix_order);
        double t1 = omp_get_wtime();
        mul_matrix(C.data(), matrix_order, matrix_order,
            A.data(), matrix_order, matrix_order,
            B.data(), matrix_order, matrix_order);
        double t2 = omp_get_wtime();
        duration += t2 - t1;

        // auto t1 = std::chrono::steady_clock::now();
        // mul_matrix(C.data(), matrix_order, matrix_order,
        //     A.data(), matrix_order, matrix_order,
        //     B.data(), matrix_order, matrix_order);
        // auto t2 = std::chrono::steady_clock::now();
        // duration += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    }
    duration /= exp_count;

    std::cout << "scalar multiplication: duration = " << duration << "s, acceleration = 1\n";
    output << "Scalar," << duration << ",1\n";

    //Vectorized matrix multiplication using AVX2. D = A*B.
    double duration_avx = 0;
    for (std::size_t i = 0; i < exp_count; i++)
    {
        randomize_matrix(A.data(), matrix_order);
        double t1 = omp_get_wtime();
        mul_matrix_avx_256(D.data(), matrix_order, matrix_order,
            A.data(), matrix_order, matrix_order,
            B.data(), matrix_order, matrix_order);
        // mul_matrix_avx_512(D.data(), matrix_order, matrix_order,
           // A.data(), matrix_order, matrix_order,
           // B.data(), matrix_order, matrix_order);
        double t2 = omp_get_wtime();
        duration_avx += t2 - t1;
    }
    duration_avx /= exp_count;

    std::cout << "vectorized multiplication: duration = " << duration_avx
        << "s, acceleration = " << duration / duration_avx << "\n";
    output << "Vectorized," << duration_avx << "," << duration / duration_avx << "\n";

    std::cout << "==Done.==\n";

    output.close();
    return 0;
}