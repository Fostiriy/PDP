#include <barrier>
#include <bit>
#include <chrono>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <thread>
#include <vector>

using namespace std;
/*
const unsigned nibbles[16] = {0, 0B1000, 0B0100, 0B1100, 0B0010, 0B1010, 0B0110, 0B1110,
                              0B0001, 0B1001, 0B0101, 0B1101, 0B0011, 0B1011, 0B0111, 0B1111};

#define bit_swop_8(b) \
        ((nibbles[(b)&0B1111] << 4 | nibbles[(b) >> 4])
#if SIZE_MAX == ~UINT64_C(0)
    #define bit_swap_size bit_swap_64
#elif SIZE_MAX == ~UINT32_C(0)
    #define bit_swap_size bit_swap_32
#else 
    #error "Unsupported" // на error ругался
#endif

constexpr size_t bit_swap(uint8_t) 
{
    
}

template <unsigned_integral T>
constexpr T bit_swap(T x) {
    if constexpr (sizeof(T) * CHAR_BIT == 64)
        return (uint64_t)bit_swap((uint64_t)x);
    else
}


void bit_shuffle(const complex<double>* inp, complex<double>* out, size_t n)
{
    size_t lzc = _Countr_zero(n);
    size_t bit_length = sizeof(size_t) * 8 - lzc - 1;

    for (size_t i = 0; i < n; n++)
    {
        size_t index = bit_swap(i);
        new_index >>= lzc;
        out[new_index] = inp[i];

    }
}*/

// Таблица для переворота битов в полубайте (4 бита)
static unsigned nibble[16] = { 0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15 };

// Функция для переворота битов в байте
unsigned flip_b(unsigned byte) {
    return (nibble[byte & 15] << 4) | nibble[(byte >> 4) & 15];
}

// Функция для переворота битов в 16-битном числе
unsigned flip_s(unsigned v) {
    return flip_b(v & 0xFF) << 8 | flip_b(v >> 8);
}

// Функция для переворота битов в 32-битном числе
unsigned flip_i(unsigned v) {
    return flip_s(v & 0xFFFF) << 16 | flip_s(v >> 16);
}

// Функция для переворота битов в 64-битном числе
unsigned long long flip_ll(unsigned long long v) {
    return (unsigned long long) flip_i(v & 0xFFFFFFFF) << 32 | flip_i(v >> 32);
}

// Функция для перестановки битов в массиве комплексных чисел
void bit_shuffle(const std::complex<double>* inp, std::complex<double>* out, std::size_t n)
{
    // Вычисление сдвига для перестановки битов
    std::size_t shift = std::countl_zero<std::size_t>(n) + 1llu;
    for (std::size_t i = 0; i < n; i++)
    {
        // Перестановка битов и копирование элемента
        out[flip_ll(i) >> shift] = inp[i];
    }
}

// Структура для хранения диапазона задач для потока
struct thread_range
{
    std::size_t b, e; // Начало и конец диапазона
};

// Функция для вычисления диапазона задач для потока
thread_range thread_task_range(std::size_t task_count, std::size_t thread_count, std::size_t thread_id)
{
    auto b = task_count % thread_count; // Вычисление начального индекса
    auto s = task_count / thread_count; // Вычисление размера задачи для каждого потока
    if (thread_id < b) b = ++s * thread_id; // Корректировка начального индекса для первых потоков
    else b += s * thread_id; // Корректировка начального индекса для остальных потоков
    size_t e = b + s; // Вычисление конечного индекса
    return { b, e }; // Возвращение диапазона задач
}

// Основная функция для выполнения FFT с использованием нескольких потоков
void fft_nonrec_multithreaded_core(const std::complex<double>* inp, std::complex<double>* out, std::size_t n, int inverse, std::size_t thread_count)
{
    // Перестановка битов в массиве комплексных чисел
    bit_shuffle(inp, out, n);

    // Создание барьера для синхронизации потоков
    std::barrier<> bar(thread_count);
    auto thread_lambda = [&out, n, inverse, thread_count, &bar](std::size_t thread_id) {
        for (std::size_t group_length = 2; group_length <= n; group_length <<= 1)
        {
            // Проверка, есть ли задачи для потока
            if (thread_id + 1 <= n / group_length)
            {
                // Вычисление диапазона задач для потока
                auto [b, e] = thread_task_range(n / group_length, thread_count, thread_id);
                for (std::size_t group = b; group < e; group++)
                {
                    for (std::size_t i = 0; i < group_length / 2; i++)
                    {
                        // Вычисление комплексного коэффициента
                        auto w = std::polar(1.0, -2 * std::numbers::pi_v<double> *i * inverse / group_length);
                        auto r1 = out[group_length * group + i];
                        auto r2 = out[group_length * group + i + group_length / 2];
                        // Обновление значений массива
                        out[group_length * group + i] = r1 + w * r2;
                        out[group_length * group + i + group_length / 2] = r1 - w * r2;
                    }
                }
            }

            // Ожидание завершения всех потоков на текущем этапе
            bar.arrive_and_wait();
        }
    };

    // Создание и запуск потоков
    std::vector<std::thread> threads(thread_count - 1);
    for (std::size_t i = 1; i < thread_count; i++)
    {
        threads[i - 1] = std::thread(thread_lambda, i);
    }
    thread_lambda(0); // Выполнение задач в основном потоке
    for (auto& i : threads)
    {
        i.join(); // Ожидание завершения всех потоков
    }
}

// Функция для выполнения прямого FFT с использованием нескольких потоков
void fft_nonrec_multithreaded(const std::complex<double>* inp, std::complex<double>* out, std::size_t n, std::size_t thread_count)
{
    fft_nonrec_multithreaded_core(inp, out, n, 1, thread_count);
}

// Функция для выполнения обратного FFT с использованием нескольких потоков
void ifft_nonrec_multithreaded(const std::complex<double>* inp, std::complex<double>* out, std::size_t n, std::size_t thread_count)
{
    fft_nonrec_multithreaded_core(inp, out, n, -1, thread_count);
    for (std::size_t i = 0; i < n; i++)
    {
        out[i] /= static_cast<std::complex<double>>(n); // Нормализация результата
    }
}

//======================================================================================

int main()
{
    const std::size_t exp_count = 10;
    constexpr std::size_t n = 1llu << 20;//25
    std::vector<std::complex<double>> original(n), spectre(n), restored(n);

    auto print_vector = [](const std::vector<std::complex<double>>& v) {
        for (std::size_t i = 0; i < v.size(); i++)
        {
            std::cout << "[" << i << "] " << std::fixed << v[i] << "\n";
        }
    };

    auto randomize_vector = [](std::vector<std::complex<double>>& v) {
        std::uniform_real_distribution<double> unif(0, 100000);
        std::default_random_engine re;
        for (std::size_t i = 0; i < v.size(); i++)
        {
            v[i] = unif(re);
        }
    };

    // Проверяет, являются ли два вектора комплексных чисел приблизительно равными
    auto approx_equal = [](const std::vector<std::complex<double>>& v,
        const std::vector<std::complex<double>>& u) {
        for (std::size_t i = 0; i < v.size(); i++)
        {
            if (std::abs(v[i] - u[i]) > 0.0001)
            {
                return false;
            }
        }
        return true;
    };

    std::ofstream output("output.csv");
    if (!output.is_open())
    {
        std::cout << "Error!\n";
        return -1;
    }
    output << "T,Duration,Acceleration\n";

    //Generate triangular signal.
    for (std::size_t i = 0; i < n / 2; i++)
    {
        original[i] = i;
        original[n - 1 - i] = i;
    }

    std::cout << "==Correctness test. ";
    fft_nonrec_multithreaded(original.data(), spectre.data(), n, 4);
    ifft_nonrec_multithreaded(spectre.data(), restored.data(), n, 4);
    if (!approx_equal(original, restored))
    {
        std::cout << "FAILURE==\n";
        return -1;
    }
    std::cout << "ok.==\n";

    std::cout << "==Performance tests.==\n";
    double time_sum_1;
    for (std::size_t i = 1; i <= std::thread::hardware_concurrency(); i++)
    {
        double time_sum = 0;

        for (std::size_t j = 0; j < exp_count; j++)
        {
            randomize_vector(original);
            auto t1 = std::chrono::steady_clock::now();
            fft_nonrec_multithreaded(original.data(), spectre.data(), n, i);
            auto t2 = std::chrono::steady_clock::now();
            time_sum += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        }

        if (i == 1)
        {
            time_sum_1 = time_sum;
        }

        std::cout << "FFT: T = " << i << ", duration = "
            << time_sum / exp_count << "ms, acceleration = " << (time_sum_1 / exp_count) / (time_sum / exp_count) << "\n";
        output << i << "," << time_sum / exp_count << "," << (time_sum_1 / exp_count) / (time_sum / exp_count) << "\n";
    }
    std::cout << "==Done.==\n";

    output.close();
    return 0;
}