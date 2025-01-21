// Лаб 1 - Расчёт интеграла функции квадрата числа

#include <iostream>
#include <chrono>
#include <omp.h>
#include <thread>
#include <fstream>
const double n = 100000000;

double f(double x) {
	return x * x;
}

// Определённый интеграл
double integrate(double a, double b) {
	double sum = 0;
	double dx = (b - a) / n;
	for (int i = 0; i < n; i++)
		sum += f(a + i * dx);
	return dx * sum;
}

// Распараллеленный определённый интегралл
double integrate_omp(double a, double b) {
	double sum = 0;
	double dx = (b - a) / n;
	
	#pragma omp parallel
	{
		double local_sum = 0;
		unsigned t = omp_get_thread_num(); // номер потока
		unsigned T = omp_get_num_threads(); // все потоки (логические ядра)
		for (size_t i = t; i < n; i += T)
			//Без мьютекса и локальных сумм несколько потоков будут
			//писать в одну переменную одновременно, что приведёт к ошибке.
			local_sum += f(a + i * dx);

		//Мьютекс из OpenMP для синхронизации потоков, записывающих
		//результат в одну переменную.
		#pragma omp critical
		{
			sum += local_sum;
		}
	}
	return dx * sum;
}


int main(int argc, char** argv)
{
	std::ofstream output("output.csv");
	if (!output.is_open())
	{
		std::cout << "Error!\n";
		return -1;
	}

	double t1 = omp_get_wtime();
	double result = integrate(-1, 1);
	double t2 = omp_get_wtime() - t1;
	std::cout << "integrate: T = 1, value = " << result << ", duration = " << t2 << "s, acceleration = 1\n";
	output << "T,Duration,Acceleration\n1," << t2 << ",1\n";

	double duration1 = t2;

	for (std::size_t i = 2; i <= std::thread::hardware_concurrency(); i++)
	{
		omp_set_num_threads(i);
		t1 = omp_get_wtime();
		result = integrate_omp(-1, 1);
		t2 = omp_get_wtime() - t1;

		std::cout << "integrate: T = " << i << ", value = " << result
			<< ", duration = " << t2 << "s, acceleration = " << duration1 / t2 << "\n";
		output << i << "," << t2 << "," << (duration1 / t2);
		
	if (i < std::thread::hardware_concurrency())
	{
		output << "\n";
	}
	}

	output.close();
}