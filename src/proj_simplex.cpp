/*==================================================================
 * Hugo Raguet 2016
 *================================================================*/
#include <alloca.h>
#include "../include/proj_simplex.hpp"
#include "../include/omp_num_threads.hpp"

/* constants of the correct type */
#define ZERO ((real_t) 0.)

#define m0 (weighted_metric ? m[0] : ((real_t) 1.))
#define md (weighted_metric ? m[d] : ((real_t) 1.))

template <typename real_t>
void proj_simplex(real_t *X, size_t D, size_t N, const real_t *A, real_t a,
    const real_t *M, const real_t *m)
{
    const bool weighted_metric = M || m;
    #pragma omp parallel firstprivate(m) NUM_THREADS(10*D*N, N)
    {
    bool *is_larger = (bool*) alloca(D*sizeof(bool));
    #pragma omp for schedule(static)
    for (size_t n = 0; n < N; n++){
        real_t *x = X + D*n;
        if (M){ m = M + D*n; };
        real_t threshold = (x[0] - (A ? A[n] : a))/m0;
        x[0] = x[0]/m0;
        is_larger[0] = true;
        real_t num_larger = m0;
        /* first pass: populate is_larger and x */
        for (size_t d = 1; d < D; d++){
            x[d] = x[d]/md;
            if (x[d] > threshold){
                is_larger[d] = true;
                num_larger += md;
                threshold += md*(x[d] - threshold)/num_larger;
            }else{
                is_larger[d] = false;
            }
        }
        /* subsequent passes */
        bool threshold_not_found = true;
        while (threshold_not_found){
            threshold_not_found = false;
            for (size_t d = 0; d < D; d++){
                if (is_larger[d]){
                    if (x[d] < threshold){
                        is_larger[d] = false;
                        num_larger -= md;
                        threshold += md*(threshold - x[d])/num_larger;
                        threshold_not_found = true;
                    }
                }
            }
        }
        /* finalize */
        for (size_t d = 0; d < D; d++){
            if (is_larger[d]){ x[d] = (x[d] - threshold)*md; }
            else{ x[d] = ZERO; }
        }
    }
    } // end omp parallel
}

/**  instantiate for compilation  **/

template void proj_simplex<float>(float*, size_t, size_t,
    const float*, float, const float*, const float*);

template void proj_simplex<double>(double*, size_t, size_t,
    const double*, double, const double*, const double*);
