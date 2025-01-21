#include "num_threads.h"
#include "vector_mod.h"
#include "mod_ops.h"

#include <thread>
#include <vector>
#include <barrier> // C++ 20

using namespace std;

IntegerWord pow_mod(IntegerWord base, IntegerWord power, IntegerWord mod) {
    IntegerWord result = 1;
    while (power > 0) {
        if (power % 2 != 0) {
            result = mul_mod(result, base, mod);
        }
        power >>= 1;
        base = mul_mod(base, base, mod);
    }
    return result;
}

IntegerWord word_pow_mod(std::size_t power, IntegerWord mod) {
    return pow_mod((-mod) % mod, power, mod);
}

struct thread_range {
    size_t b, e;
};

thread_range vector_thread_range(size_t N, size_t T, size_t t) {    
    auto b = N % T;
    auto s = N / T;

    if (t < b) 
        b = ++s * t;
    else 
        b = s * t + b;

    size_t e = b + s;

    return thread_range(b, e);
}

// Защита от ложного выделения
struct partial_result_t {
    alignas(std::hardware_destructive_interference_size) IntegerWord value;
};

IntegerWord vector_mod(const IntegerWord* V, size_t N, IntegerWord mod){
    unsigned T = get_num_threads();
    vector<thread> threads(T-1);
    vector<partial_result_t> partial_results(T); //учитываем, что главный поток выполняет лямбду с t = 0
    IntegerWord S = 0;
    barrier<> bar(T);

    auto thread_lambda = [V, N, T, mod, &partial_results, &bar](unsigned t) 
    {
        auto [b, e] = vector_thread_range(N, T, t);

        IntegerWord sum = 0;
        // По схеме Хорнера
        for (unsigned i = e; b < i;) // 0 < i
           //sum = (sum * x + V[e-1-i]) % mod;
            sum = add_mod(times_word(sum, mod), V[--i], mod); // то же самое, но без переполнения

        // ... код про общую сумму
        partial_results[t].value = sum;
        for (size_t i = 1, ii = 2; i < T; i = ii, ii += ii) {
            bar.arrive_and_wait();
            if (t % ii == 0 && t + i < T) {
                auto neighbor = vector_thread_range(N, T, t + i);
                partial_results[t].value = add_mod(partial_results[t].value,
                    mul_mod(partial_results[t + i].value, word_pow_mod(neighbor.b - b, mod), mod),
                    mod);
            }
        }
    };

    for (size_t i = 1; i < T; ++i) {
        threads[i - 1] = thread(thread_lambda, i);
    }
    thread_lambda(0);

    for (auto& i : threads) 
        i.join();

    return partial_results[0].value;
}