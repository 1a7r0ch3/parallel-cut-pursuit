/*==========================  omp_num_threads.hpp  ============================
 * include openmp and provide a function to compute a smart number of threads
 *
 * Hugo Raguet 2018
 *===========================================================================*/
#pragma once
#ifdef _OPENMP
    #include <omp.h>
    /* rough minimum number of operations per thread */
    #define MIN_OPS_PER_THREAD 1000
    #include <cstdint>  // requires C++11, needed for uintmax_t
#else /* provide default definitions for basic openmp queries */
static inline int omp_get_num_procs(){ return 1; }
static inline int omp_get_thread_num(){ return 0; }
#endif

/* num_ops is a rough estimation of the total number of operations 
 * max_threads is the maximum number of jobs performed in parallel */
static inline int compute_num_threads(uintmax_t num_ops, uintmax_t max_threads)
{
#ifdef _OPENMP
    const int m = (omp_get_num_procs() < max_threads) ?
                   omp_get_num_procs() : max_threads;
    int n = (num_ops > MIN_OPS_PER_THREAD) ?
             num_ops/MIN_OPS_PER_THREAD : 1;
    return (n < m) ? n : m;
#else
    return 1;
#endif
}

#define NUM_THREADS(...) num_threads(compute_num_threads((uintmax_t) __VA_ARGS__))

/* overload for max_threads defaulting to num_ops */
static inline int compute_num_threads(uintmax_t num_ops)
{ return compute_num_threads(num_ops, num_ops); }
