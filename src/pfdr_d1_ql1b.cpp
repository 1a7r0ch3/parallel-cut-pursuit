/*=============================================================================
 * Hugo Raguet 2016
 *===========================================================================*/
#include <cmath>
#include "../include/pfdr_d1_ql1b.hpp"
#include "../include/matrix_tools.hpp"
#include "../include/omp_num_threads.hpp"

/* constants of the correct type */
#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)
#define HALF ((real_t) 0.5)
#define INF_REAL (std::numeric_limits<real_t>::infinity())
#define Y_(n) (Y ? Y[(n)] : (real_t) 0.0)
#define Yl1_(v) (Yl1 ? Yl1[(v)] : (real_t) 0.0)

using namespace std;

template <typename real_t, typename vertex_t>
Pfdr_d1_ql1b<real_t, vertex_t>::Pfdr_d1_ql1b(vertex_t V, size_t E,
    const vertex_t* edges) : Pfdr_d1<real_t, vertex_t>(V, E, edges)
{
    /* ensure handling of infinite values (negation, comparisons) is safe */
    static_assert(numeric_limits<real_t>::is_iec559,
        "PFDR d1 quadratic l1 bounds: real_t must satisfy IEEE 754.");
    Y = Yl1 = A = R = nullptr;
    N = DIAG_ATA;
    a = ONE;
    l1_weights = nullptr; homo_l1_weight = ZERO;
    low_bnd = nullptr; homo_low_bnd = -INF_REAL;
    upp_bnd = nullptr; homo_upp_bnd = INF_REAL;

    lipsch_equi = JACOBI;
    lipsch_norm_tol = 1e-3;
    lipsch_norm_it_max = 100;
    lipsch_norm_nb_init = 10;
}

template <typename real_t, typename vertex_t>
Pfdr_d1_ql1b<real_t, vertex_t>::~Pfdr_d1_ql1b(){ free(R); }

template <typename real_t, typename vertex_t>
void Pfdr_d1_ql1b<real_t, vertex_t>::set_lipsch_norm_param(
    Equilibration lipsch_equi, real_t lipsch_norm_tol,
    int lipsch_norm_it_max, int lipsch_norm_nb_init)
{
    this->lipsch_equi = lipsch_equi;
    this->lipsch_norm_tol = lipsch_norm_tol;
    this->lipsch_norm_it_max = lipsch_norm_it_max;
    this->lipsch_norm_nb_init = lipsch_norm_nb_init;
}

template <typename real_t, typename vertex_t>
void Pfdr_d1_ql1b<real_t, vertex_t>::set_quadratic(const real_t* Y,
    size_t N, const real_t* A, real_t a)
{
    if (!A && !a){ N = DIAG_ATA; } // no quadratic part !
    free(R);
    R = IS_ATA(N) ? nullptr : (real_t*) malloc_check(sizeof(real_t)*N);
    this->Y = Y; this->N = N; this->A = A; this->a = a;
}

template <typename real_t, typename vertex_t>
void Pfdr_d1_ql1b<real_t, vertex_t>::set_l1(const real_t* l1_weights,
    real_t homo_l1_weight, const real_t* Yl1)
{
    if (!l1_weights && homo_l1_weight < ZERO){
        cerr << "PFDR graph d1 quadratic l1 bounds: negative homogeneous l1 "
            "penalization (" << homo_l1_weight << ")." << endl;
        exit(EXIT_FAILURE);
    }
    this->l1_weights = l1_weights; this->homo_l1_weight = homo_l1_weight;
    this->Yl1 = Yl1;
}

template <typename real_t, typename vertex_t>
void Pfdr_d1_ql1b<real_t, vertex_t>::set_bounds(
    const real_t* low_bnd, real_t homo_low_bnd,
    const real_t* upp_bnd, real_t homo_upp_bnd)
{
    if (!low_bnd && !upp_bnd && homo_low_bnd > homo_upp_bnd){
        cerr << "PFDR graph d1 quadratic l1 bounds: homogeneous lower bound ("
            << homo_low_bnd << ") greater than homogeneous upper bound ("
            << homo_upp_bnd << ")." << endl;
        exit(EXIT_FAILURE);
    }
    this->low_bnd = low_bnd; this->homo_low_bnd = homo_low_bnd;
    this->upp_bnd = upp_bnd; this->homo_upp_bnd = homo_upp_bnd;
}

template <typename real_t, typename vertex_t>
void Pfdr_d1_ql1b<real_t, vertex_t>::apply_A()
{
    if (!IS_ATA(N)){ /* direct matricial case, compute residual R = Y - A X */
        #pragma omp parallel for schedule(static) NUM_THREADS(N*V, N)
        for (size_t n = 0; n < N; n++){
            R[n] = Y_(n);
            size_t i = n;
            for (vertex_t v = 0; v < V; v++){
                R[n] -= A[i]*X[v];
                i += N;
            }
        }
    }else if (N == FULL_ATA){ /* premultiplied by A^t, compute (A^t A) X */
        #pragma omp parallel for schedule(static) NUM_THREADS(V*V, V)
        for (vertex_t v = 0; v < V; v++){
            const real_t *Av = A + V*v;
            AX[v] = ZERO;
            for (vertex_t u = 0; u < V; u++){ AX[v] += Av[u]*X[u]; }
        }
    }else if (A){ /* diagonal case, compute (A^t A) X */
        #pragma omp parallel for schedule(static) NUM_THREADS(V)
        for (vertex_t v = 0; v < V; v++){ AX[v] = A[v]*X[v]; }
    }else if (a){ /* identity matrix */
        for (vertex_t v = 0; v < V; v++){ AX[v] = X[v]; }
    }
}

template<typename real_t, typename vertex_t>
void Pfdr_d1_ql1b<real_t, vertex_t>::compute_lipschitz_metric()
{
    if (N == DIAG_ATA){ /* diagonal case */
        if (A){
            L = A;
            lshape = MONODIM;
        }else if (a){ /* identity matrix */
            l = ONE;
            lshape = SCALAR;
        }else{ /* no quadratic penalty */
            l = ZERO;
            lshape = SCALAR;
        }
    }else if (lipsch_equi == NOEQUI){
        l = operator_norm_matrix(N, V, A, (const real_t*) nullptr,
            lipsch_norm_tol, lipsch_norm_it_max, lipsch_norm_nb_init);
        lshape = SCALAR;
    }else{
        Lmut = (real_t*) malloc_check(V*sizeof(real_t));
        switch (lipsch_equi){
        case JACOBI:
            symmetric_equilibration_jacobi<real_t>(N, V, A, Lmut); break;
        case BUNCH:
            symmetric_equilibration_bunch<real_t>(N, V, A, Lmut); break;
        }

        /* stability: ratio between two elements no more than cond_min;
         * outliers are expected to be in the high range, hence the minimum
         * is used for the base reference */
        real_t lmin = Lmut[0];
        #pragma omp parallel for schedule(static) NUM_THREADS(V) \
            reduction(min:lmin)
        for (vertex_t v = 1; v < V; v++){
            if (Lmut[v] < lmin){ lmin = Lmut[v]; }
        }
        real_t lmax = lmin/cond_min;
        #pragma omp parallel for schedule(static) NUM_THREADS(V)
        for (vertex_t v = 0; v < V; v++){
            if (Lmut[v] > lmax){ Lmut[v] = lmax; }
        }

        /* norm of the equilibrated matrix and final Lipschitz norm */
        l = operator_norm_matrix(N, V, A, Lmut, lipsch_norm_tol, 
            lipsch_norm_it_max, lipsch_norm_nb_init);
        #pragma omp parallel for schedule(static) NUM_THREADS(2*V, V)
        for (vertex_t v = 0; v < V; v++){ Lmut[v] = l/(Lmut[v]*Lmut[v]); }
        L = Lmut;
        lshape = MONODIM;
    }
}

template<typename real_t, typename vertex_t>
void Pfdr_d1_ql1b<real_t, vertex_t>::compute_hess_f()
{ for (vertex_t v = 0; v < V; v++){ Ga[v] = L ? L[v] : l; } }

template<typename real_t, typename vertex_t>
void Pfdr_d1_ql1b<real_t, vertex_t>::add_pseudo_hess_h()
/* l1 contribution
 * a local quadratic approximation of x -> ||x|| at z is
 * x -> 1/2 (||x||^2/||z|| + ||z||)
 * whose second order derivative is 1/||z|| */
{
    if (l1_weights || homo_l1_weight){
        #pragma omp parallel for schedule(static) NUM_THREADS(3*V, V)
        for (vertex_t v = 0; v < V; v++){
            real_t amp = abs(X[v] - Yl1_(v));
            if (amp < eps){ amp = eps; }
            Ga[v] += l1_weights ? l1_weights[v]/amp : homo_l1_weight/amp;
        }
    }
}

template <typename real_t, typename vertex_t>
void Pfdr_d1_ql1b<real_t, vertex_t>::compute_Ga_grad_f()
/* supposed to be called after apply_A() */
{
    if (!IS_ATA(N)){ /* direct matricial case, grad = -(A^t) R */
        #pragma omp parallel for schedule(static) NUM_THREADS(V*N, V)
        for (vertex_t v = 0; v < V; v++){
            const real_t *Av = A + N*v;
            Ga_grad_f[v] = ZERO;
            for (size_t n = 0; n < N; n++){ Ga_grad_f[v] -= Av[n]*R[n]; }
            Ga_grad_f[v] *= Ga[v];
        }
    }else if (A || a){ /* premultiplied by A^t, grad = (A^t A) X - A^t Y */
        #pragma omp parallel for schedule(static) NUM_THREADS(V)
        for (vertex_t v = 0; v < V; v++){
            Ga_grad_f[v] = Ga[v]*(AX[v] - Y_(v));
        }
    }else{ /* no quadratic part */
        for (vertex_t v = 0; v < V; v++){ Ga_grad_f[v] = ZERO; }
    }
}

template<typename real_t, typename vertex_t>
void Pfdr_d1_ql1b<real_t, vertex_t>::compute_prox_Ga_h()
{
    #pragma omp parallel for schedule(static) NUM_THREADS(V)
    for (vertex_t v = 0; v < V; v++){
        if (l1_weights || homo_l1_weight){
            real_t th_l1 = (l1_weights ? l1_weights[v] : homo_l1_weight)*Ga[v];
            real_t dif = X[v] - Yl1_(v);
            if (dif > th_l1){ dif -= th_l1; }
            else if (dif < -th_l1){ dif += th_l1; }
            else{ dif = ZERO; }
            X[v] = Yl1_(v) + dif;
        }
        if (low_bnd){
            if (X[v] < low_bnd[v]){ X[v] = low_bnd[v]; }
        }else if (homo_low_bnd > -INF_REAL){
            if (X[v] < homo_low_bnd){ X[v] = homo_low_bnd; }
        }
        if (upp_bnd){
            if (X[v] > upp_bnd[v]){ X[v] = upp_bnd[v]; }
        }else if (homo_upp_bnd < INF_REAL){
            if (X[v] > homo_upp_bnd){ X[v] = homo_upp_bnd; }
        }
    }
}

template<typename real_t, typename vertex_t>
real_t Pfdr_d1_ql1b<real_t, vertex_t>::compute_f()
{
    real_t obj = ZERO;
    if (!IS_ATA(N)){ /* direct matricial case, 1/2 ||Y - A X||^2 */
        #pragma omp parallel for schedule(static) NUM_THREADS(N) \
            reduction(+:obj)
        for (size_t n = 0; n < N; n++){ obj += R[n]*R[n]; }
        obj *= HALF;
    }else if (A || a){ /* premultiplied by A^t, 1/2 <X, A^t A X> - <X, A^t Y> */
        #pragma omp parallel for schedule(static) NUM_THREADS(V) \
            reduction(+:obj)
        for (vertex_t v = 0; v < V; v++){
            obj += X[v]*(HALF*AX[v] - Y_(v));
        }
    }
    return obj;
}

template<typename real_t, typename vertex_t>
real_t Pfdr_d1_ql1b<real_t, vertex_t>::compute_h()
{
    real_t obj = ZERO;
    if (l1_weights || homo_l1_weight){ /* ||x||_l1 */
        #pragma omp parallel for schedule(static) NUM_THREADS(V) \
             reduction(+:obj)
        for (vertex_t v = 0; v < V; v++){
            if (l1_weights){ obj += l1_weights[v]*abs(X[v] - Yl1_(v)); }
            else{ obj += homo_l1_weight*abs(X[v] - Yl1_(v)); }
        }
    }
    return obj;
}

template<typename real_t, typename vertex_t>
void Pfdr_d1_ql1b<real_t, vertex_t>::initialize_iterate()
/* initialize with coordinatewise pseudo-inverse pinv = <Av, Y>/||Av||^2,
 * or on l1 target if there is no quadratic part */
{
    if (!X){ X = (real_t*) malloc_check(sizeof(real_t)*V); }

    /* prevent useless computations */
    if (A && !Y){
        for (vertex_t v = 0; v < V; v++){ X[v] = ZERO; }
        return;
    }

    if (IS_ATA(N)){ /* left-premultiplied by A^t case */
        if (A){
            size_t Vdiag = N == FULL_ATA ? V + 1 : 1;
            #pragma omp parallel for schedule(static) NUM_THREADS(V)
            for (vertex_t v = 0; v < V; v++){
                X[v] = A[Vdiag*v] > ZERO ? Y[v]/A[Vdiag*v] : ZERO;
            }
        }else if (a){ /* identity */
            for (vertex_t v = 0; v < V; v++){ X[v] = Y_(v); }
        }else{ /* no quadratic part, initialize on l1 */
            for (vertex_t v = 0; v < V; v++){ X[v] = Yl1_(v); }
        }
    }else{ /* direct matricial case */
        #pragma omp parallel for schedule(static) NUM_THREADS(2*N*V, V)
        for (vertex_t v = 0; v < V; v++){
            const real_t* Av = A + N*v;
            real_t AvY = ZERO;
            real_t Av2 = ZERO;
            for (size_t n = 0; n < N; n++){
                AvY += Av[n]*Y[n];
                Av2 += Av[n]*Av[n];
            }
            X[v] = Av2 > ZERO ? AvY/Av2 : ZERO;
        }
    }
}

template<typename real_t, typename vertex_t>
void Pfdr_d1_ql1b<real_t, vertex_t>::preconditioning(bool init)
{
    Pfdr_d1<real_t, vertex_t>::preconditioning(init);

    if (init){ /* reinitialize according to penalizations */
        vertex_t num_ops = (low_bnd || homo_low_bnd > -INF_REAL ||
            upp_bnd || homo_upp_bnd < INF_REAL) ? V : 1;
        #pragma omp parallel for schedule(static) NUM_THREADS(num_ops)
        for (vertex_t v = 0; v < V; v++){
            if (l1_weights || homo_l1_weight){ X[v] = Yl1_(v); } /* sparsity */
            if (low_bnd){
                if (X[v] < low_bnd[v]){ X[v] = low_bnd[v]; }
            }else if (homo_low_bnd > -INF_REAL){
                if (X[v] < homo_low_bnd){ X[v] = homo_low_bnd; }
            }
            if (upp_bnd){
                if (X[v] > upp_bnd[v]){ X[v] = upp_bnd[v]; }
            }else if (homo_upp_bnd < INF_REAL){
                if (X[v] > homo_upp_bnd){ X[v] = homo_upp_bnd; }
            }
        }
        initialize_auxiliary();
    }

    apply_A();
}

template<typename real_t, typename vertex_t>
void Pfdr_d1_ql1b<real_t, vertex_t>::main_iteration()
{
    Pfdr<real_t, vertex_t>::main_iteration();
    
    apply_A();
}

/**  instantiate for compilation  **/
template class Pfdr_d1_ql1b<float, uint16_t>;

template class Pfdr_d1_ql1b<float, uint32_t>;

template class Pfdr_d1_ql1b<double, uint16_t>;

template class Pfdr_d1_ql1b<double, uint32_t>;
