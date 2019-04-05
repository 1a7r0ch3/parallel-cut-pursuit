/*=============================================================================
 * compute the squared operator norm (squared greatest eigenvalue) of a real 
 * matrix using power method.
 *
 * Hugo Raguet 2016
 *===========================================================================*/
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "../include/omp_num_threads.hpp"
#include "../include/matrix_tools.hpp"

/* constants of the correct type */
#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)

using namespace std;

static const int HALF_RAND_MAX = (RAND_MAX/2 + 1);
static const double HALF_RAND_MAX_D = (double) HALF_RAND_MAX;

template <typename real_t>
real_t compute_norm(const real_t *X, const size_t N)
{
    real_t norm = ZERO;
    for (size_t n = 0; n < N; n++){ norm += X[n]*X[n]; }
    return sqrt(norm);
}

template <typename real_t>
void normalize_and_apply_matrix(const real_t* A, real_t* X, real_t* AX,
    const real_t* D, real_t norm, bool sym, size_t M, size_t N)
{
    if (sym){
        if (D){ for (size_t n = 0; n < N; n++){ AX[n] = X[n]*D[n]/norm; } }
        else{ for (size_t n = 0; n < N; n++){ AX[n] = X[n]/norm; } }
    }else{
        if (D){ for (size_t n = 0; n < N; n++){ X[n] *= D[n]/norm; } }
        else{ for (size_t n = 0; n < N; n++){ X[n] /= norm; } }
        /* apply A */
        for (size_t m = 0; m < M; m++){
            AX[m] = ZERO;
            size_t p = m;
            for (size_t n = 0; n < N; n++){
                AX[m] += A[p]*X[n];
                p += M;
            }
        }
    }
    /* apply A^t or AA */
    const real_t *An = A;
    for (size_t n = 0; n < N; n++){
        X[n] = ZERO;
        for (size_t m = 0; m < M; m++){ X[n] += An[m]*AX[m]; }
        An += M;
    }
    if (D){ for (size_t n = 0; n < N; n++){ X[n] *= D[n]; } }
}

template <typename real_t>
real_t operator_norm_matrix(size_t M, size_t N, const real_t* A,
    const real_t* D, real_t tol, int it_max, int nb_init, bool verbose)
{
    real_t* AA = nullptr;
    bool sym = false;

    /**  preprocessing  **/
    const size_t P = (M < N) ? M : N;
    const int i_tot = nb_init*it_max;
    if (P == FULL_ATA){
        sym = true;
        M = (M > N) ? M : N;
        N = M;
    }else if (2*M*N*i_tot > (M*N*P + P*P*i_tot)){
        sym = true;
        /* compute symmetrization */
        AA = (real_t*) malloc(sizeof(real_t)*P*P);
        if (!AA){
            cerr << "Operator norm matrix: not enough memory." << endl;
            exit(EXIT_FAILURE);
        }
        for (size_t p = 0; p < P*P; p++){ AA[p] = ZERO; }
        if (M < N){ /* A A^t is smaller */
            /* fill upper triangular part (from lower triangular products) */
            #pragma omp parallel for schedule(static) NUM_THREADS(M*N*P/2, P)
            for (size_t p = 0; p < P; p++){
                const real_t *Ap = A + p; // run along p-th row of A
                const real_t *An = A; // n-th row of A^t
                real_t *AAp = AA + P*p; // p-th column of AA
                real_t ApnDn2;
                for (size_t n = 0; n < N; n++){
                    if (D){
                        ApnDn2 = (*Ap)*D[n]*D[n];
                        for (size_t m = 0; m <= p; m++){
                            AAp[m] += ApnDn2*An[m];
                        }
                    }else{
                        for (size_t m = 0; m <= p; m++){
                            AAp[m] += (*Ap)*An[m];
                        }
                    }
                    An += M;
                    Ap += M;
                }
            }
        }else{ /* A^t A is smaller */
            /* fill upper triangular part */
            #pragma omp parallel for schedule(static) NUM_THREADS(M*N*P/2, P)
            for (size_t p = 0; p < P; p++){
                const real_t *Ap = A + M*p; // p-th column of A 
                const real_t *An = A; // run along n-th column of A
                real_t *AAp = AA + P*p; // p-th column of AA
                for (size_t n = 0; n <= p; n++){
                    AAp[n] = ZERO;
                    for (size_t m = 0; m < M; m++){ AAp[n] += (*(An++))*Ap[m]; }
                    if (D){ AAp[n] *= D[p]*D[n]; }
                }
            }
        }
        /* fill lower triangular part */
        #pragma omp parallel for schedule(static) NUM_THREADS(P, P - 1)
        for (size_t p = 0; p < P - 1; p++){
            real_t *AAp = AA + P*p;
            size_t m = (P + 1)*p + P;
            for (size_t n = p + 1; n < P; n++){
                AAp[n] = AA[m];
                m += P;
            }
        }
        M = P;
        N = P;
        A = AA;
        /* D has been taken into account in A D^2 A^t or D A^t A D */
        if (D){ D = nullptr; }
    }

    /**  power method  **/
    const int num_procs = omp_get_num_procs();
    nb_init = (1 + (nb_init - 1)/num_procs)*num_procs;
    if (verbose){
        cout << "compute matrix operator norm on " << nb_init << " random "
            << "initializations, over " << num_procs << " parallel threads... "
            << flush;
    }

    real_t matrix_norm2 = ZERO;
    #pragma omp parallel reduction(max:matrix_norm2) num_threads(num_procs)
    {
    unsigned int rand_seed = time(nullptr) + omp_get_thread_num();
    real_t *X = (real_t*) alloca(N*sizeof(real_t));
    real_t *AX = (real_t*) alloca(M*sizeof(real_t));
    #pragma omp for schedule(static)
    for (int init = 0; init < nb_init; init++){
        /* random initialization */
        for (size_t n = 0; n < N; n++){
            /* very crude uniform distribution on [-1,1] */
            X[n] = (rand_r(&rand_seed) - HALF_RAND_MAX)/HALF_RAND_MAX_D;
        }
        real_t norm = compute_norm(X, N);
        normalize_and_apply_matrix(A, X, AX, D, norm, sym, M, N);
        norm = compute_norm(X, N);
        /* iterate */
        if (norm > ZERO){
            for (int it = 0; it < it_max; it++){
                normalize_and_apply_matrix(A, X, AX, D, norm, sym, M, N);
                real_t norm_ = compute_norm(X, N);
                if ((norm_ - norm)/norm < tol){ break; }
                norm = norm_;
            }
        }
        if (norm > matrix_norm2){ matrix_norm2 = norm; }
    }
    } // end pragma omp parallel
    if (verbose){ cout << "done." << endl; }
    free(AA);
    return matrix_norm2;
}

template <typename real_t>
void symmetric_equilibration_jacobi(size_t M, size_t N, const real_t* A,
    real_t* D)
{
    if (M == FULL_ATA){ /* premultiplied by A^t */
        #pragma omp parallel for schedule(static) NUM_THREADS(N)
        for (size_t n = 0; n < N; n++){ D[n] = ONE/sqrt(A[n*(N + 1)]); }
    }else{
        #pragma omp parallel for schedule(static) NUM_THREADS(M*N, N)
        for (size_t n = 0; n < N; n++){
            const real_t *An = A + M*n;
            D[n] = ZERO;
            for (size_t m = 0; m < M; m++){ D[n] += An[m]*An[m]; }
            D[n] = ONE/sqrt(D[n]);
        }
    }
}

template <typename real_t>
void symmetric_equilibration_bunch(size_t M, size_t N, const real_t* A,
    real_t* D)
{
    if (M == FULL_ATA){ /* premultiplied by A^t */
        D[0] = ONE/sqrt(A[0]);
    }else{
        real_t A1A1 = ZERO;
        #pragma omp parallel for NUM_THREADS(M) reduction(+:A1A1)
        for (size_t m = 0; m < M; m++){ A1A1 += A[m]*A[m]; }
        D[0] = ONE/sqrt(A1A1);
    }

    for (size_t i = 1; i < N; i++){ 
        real_t invDi = ZERO;
        if (M == FULL_ATA){
            #pragma omp parallel for NUM_THREADS(i + 1) reduction(max:invDi)
            for (size_t j = 0; j <= i; j++){
                real_t DjAiAj = A[i + N*j];
                DjAiAj = (j < i) ? abs(DjAiAj)*D[j] : sqrt(DjAiAj);
                if (DjAiAj > invDi){ invDi = DjAiAj; }
            }
        }else{
            const real_t* Ai = A + M*i;
            #pragma omp parallel for NUM_THREADS((i + 1)*M, i + 1) \
                reduction(max:invDi)
            for (size_t j = 0; j <= i; j++){
                real_t DjAiAj = ZERO;
                const real_t* Aj = A + M*j;
                for (size_t m = 0; m < M; m++){ DjAiAj += Ai[m]*Aj[m]; }
                DjAiAj = (j < i) ? abs(DjAiAj)*D[j] : sqrt(DjAiAj);
                if (DjAiAj > invDi){ invDi = DjAiAj; }
            }
        }
        D[i] = ONE/invDi;
    }
}

/* instantiate for compilation */
template float operator_norm_matrix<float>(size_t, size_t, const float*,
    const float*, float, int, int, bool);

template double operator_norm_matrix<double>(size_t, size_t, const double*,
    const double*, double, int, int, bool);

template void symmetric_equilibration_jacobi<float>(size_t M, size_t N,
    const float* A, float* L);

template void symmetric_equilibration_jacobi<double>(size_t M, size_t N,
    const double* A, double* L);

template void symmetric_equilibration_bunch<float>(size_t M, size_t N,
    const float* A, float* L);

template void symmetric_equilibration_bunch<double>(size_t M, size_t N,
    const double* A, double* L);
