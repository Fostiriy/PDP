// Лаб 2 — сложение матриц

#include <iostream>
#include <algorithm>
#include <omp.h>
#include <cstring>
#include <immintrin.h>
#include <fstream>
#include <vector>
#include <chrono>

#define Cols 2048 * 2
#define Rows 2048 * 2

// Классическое сложение матриц
void add_matrix(double* A, const double* B, const double* C, size_t cols, size_t rows) {
    for (size_t i = 0; i < cols * rows; i++) {
		A[i] = B[i] + C[i];
	}
}

// Распараллеленное сложение матриц
void add_matrix_256(double* A, const double* B, const double* C, size_t cols, size_t rows) {
    // Берём шаг 4, так как 256 / 64 = 4
    for (size_t i = 0; i < rows * cols / 4; i++) {
		__m256d b = _mm256_loadu_pd(&(B[i * 4]));
		__m256d c = _mm256_loadu_pd(&(C[i * 4]));
		__m256d a = _mm256_add_pd(b, c);

		_mm256_storeu_pd(&(A[i * 4]), a);
	}
}

void add_matrix_512(double* A, const double* B, const double* C, size_t cols, size_t rows) {
    // Берём шаг 8, так как 512 / 64 = 8
    for (size_t i = 0; i < rows * cols / 8; i++) {
		__m512d b = _mm512_loadu_pd(&(B[i * 8]));
		__m512d c = _mm512_loadu_pd(&(C[i * 8]));
		__m512d a = _mm512_add_pd(b, c);

		_mm512_storeu_pd(&(A[i * 8]), a);
	}
}

int main(int argc, char** argv) {
	const std::size_t exp_count = 10;

	std::ofstream output("output.csv");
	if (!output.is_open())
	{
		std::cout << "Error!\n";
		return -1;
	}

	std::vector<double> B(Cols * Rows, 1), C(Cols * Rows, -1), A(Cols * Rows, 4);

	auto show_matrix = [](const double* A, size_t colsc, size_t rowsc)
	{
		for (size_t r = 0; r < rowsc; ++r)
		{
			std::cout << "[" << A[r + 0 * rowsc];
			for (size_t c = 1; c < colsc; ++c)
			{
				std::cout << ", " << A[r + c * rowsc];
			}
			std::cout << "]\n";
		}
		std::cout << "\n";
	};

	std::cout << "==Correctness test. ";
	add_matrix_256(A.data(), B.data(), C.data(), Cols, Rows);
	for (std::size_t i = 0; i < Cols * Rows; i++)
	{
		if (A.data()[i])
		{
			std::cout << "FAILURE==\n";
			return -1;
		}
	}
	std::cout << "ok.==\n";

	std::cout << "==Performance tests.==\n";
	output << "Type,Duration,Acceleration\n";

	//Scalar addition. A = B + C.
	double duration = 0;
	for (std::size_t i = 0; i < exp_count; i++) {
		double t1 = omp_get_wtime();
		add_matrix(A.data(), B.data(), C.data(), Cols, Rows);
		double t2 = omp_get_wtime();
		duration += t2 - t1;
		// auto t1 = std::chrono::steady_clock::now();
		// add_matrix(A.data(), B.data(), C.data(), cols, rows);
		// auto t2 = std::chrono::steady_clock::now();
		// duration += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	}
	duration /= exp_count;

	std::cout << "scalar addition: duration = " << duration << "s, acceleration = 1\n";
	output << "Scalar," << duration << ",1\n";

	//An attempt to destroy CPU cache.
	//std::fill_n(A.data(), rows * cols, 0);

	//Vectorized matrix addition using AVX2. A = B + C.
	double duration_avx = 0;
	for (std::size_t i = 0; i < exp_count; i++)
	{
		auto t1 = omp_get_wtime();
		add_matrix_256(A.data(), B.data(), C.data(), Cols, Rows);
		// add_matrix_512(A.data(), B.data(), C.data(), Cols, Rows);
		auto t2 = omp_get_wtime();
		duration_avx += t2 - t1;
	}
	duration_avx /= exp_count;

	std::cout << "vectorized addition: duration = " << duration_avx
		<< "s, acceleration = " << duration / duration_avx << "\n";
	output << "Vectorized," << duration_avx << "," << duration / duration_avx << "\n";

	std::cout << "==Done.==\n";

	output.close();
	//_CrtDumpMemoryLeaks();
	return 0;
}