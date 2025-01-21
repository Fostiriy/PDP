#include "vector_mod.h"
#include "test.h"
#include "performance.h"
#include <iostream>
#include <iomanip>
#include "num_threads.h"
#include <fstream>

int main(int argc, char** argv)
{
	// Тесты корректности
	std::cout << "==Correctness tests. ";
	for (std::size_t iTest = 0; iTest < test_data_count; ++iTest)
	{
		// Сравнение ожидаемого результата с результатом функции vector_mod: проверяются результаты выполнения функции для заранее заданных данныхт(test_data)
		if (test_data[iTest].result != vector_mod(test_data[iTest].dividend, test_data[iTest].dividend_size, test_data[iTest].divisor))
		{
			std::cout << "FAILURE==\n";
			return -1;
		}
	}
	std::cout << "ok.==\n";

	// Тесты производительности
	std::cout << "==Performance tests. ";
	auto measurements = run_experiments(); // Запускается алгоритм с различным количеством потоков. Для каждого измеряется время выполнения и рассчитывается ускорение относительно однопоточного режима
	std::cout << "Done==\n";

	// Вывод результатов
	std::cout << std::setfill(' ') << std::setw(2) << "T:" << " |"
		<< std::setw(3 + 2 * sizeof(IntegerWord)) << "Value:" << " | "
		<< std::setw(14) << "Duration, ms:" << " | Acceleration:\n";

	// Создание файла для записи результатов
	std::ofstream output("output.csv");
	if (!output.is_open()) {
		std::cerr << "Error: Unable to open file for writing results.\n";
		return -1;
	}

	output << "T,Value,Duration(ms),Acceleration\n"; // Заголовки файла

	// Цикл для вывода результатов экспериментов
	for (std::size_t T = 1; T <= measurements.size(); ++T)
	{
		// Вывод в консоль результатов: номер потока, значение результата, время выполнения, ускорение
		std::cout << std::setw(2) << T << " | 0x" << std::setw(2 * sizeof(IntegerWord)) << std::setfill('0') << std::hex << measurements[T - 1].result;
		std::cout << " | " << std::setfill(' ') << std::setw(14) << std::dec << measurements[T - 1].time.count();
		std::cout << " | " << (static_cast<double>(measurements[0].time.count()) / measurements[T - 1].time.count()) << "\n";

		// Запись результатов в файл
		output << T << ",0x" << std::setw(2 * sizeof(IntegerWord)) << std::setfill('0') << std::hex << measurements[T - 1].result << ",";
		output << std::dec << measurements[T - 1].time.count() << ",";
		output << (static_cast<double>(measurements[0].time.count()) / measurements[T - 1].time.count()) << "\n";

	}

	output.close();
	std::cout << "Results written to file: output.csv\n";

	return 0;
}