#include "num_threads.h"
#include <omp.h> //MSVC: /openmp, gcc: -fopenmp
#include <thread>

static unsigned thread_num = std::thread::hardware_concurrency(); // можно подставить 1

EXTERN_C void set_num_threads(unsigned T)
{
    if (!T || T > (unsigned) omp_get_num_procs())
        T = (unsigned) omp_get_num_procs();
    thread_num = T;
    omp_set_num_threads((int) T);
}

EXTERN_C unsigned get_num_threads() {
    return thread_num;
}
