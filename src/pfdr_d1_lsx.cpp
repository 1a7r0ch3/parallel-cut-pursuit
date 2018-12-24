/*=============================================================================
 * Hugo Raguet 2016, 2018
 *===========================================================================*/
#include <cmath>
#include "../include/pfdr_d1_lsx.hpp"
#include "../include/proj_simplex.hpp"
#include "../include/omp_num_threads.hpp"

/* constants of the correct type */
#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)
#define HALF ((real_t) 0.5)
#define INF_REAL (std::numeric_limits<real_t>::infinity())

#define LOSS_WEIGHTS_(v) (loss_weights ? loss_weights[(v)] : ONE)
#define Ga_(v, vd)   (gashape == MONODIM ? Ga[(v)] : Ga[(vd)])

using namespace std;

template <typename real_t, typename vertex_t>
Pfdr_d1_lsx<real_t, vertex_t>::Pfdr_d1_lsx(vertex_t V, size_t D, size_t E,
    const vertex_t* edges, real_t loss, const real_t* Y,
    const real_t* d1_coor_weights) :
    Pfdr_d1<real_t, vertex_t>(V, D, E, edges, D11, d1_coor_weights, 
        loss == LINEAR ? NULH : loss == QUADRATIC ? MONODIM : MULTIDIM),
    loss(loss), Y(Y)
{
    KL_Ga_Y = nullptr;
    loss_weights = nullptr;
}

template <typename real_t, typename vertex_t>
Pfdr_d1_lsx<real_t, vertex_t>::~Pfdr_d1_lsx(){ free(KL_Ga_Y); }

template <typename real_t, typename vertex_t>
void Pfdr_d1_lsx<real_t, vertex_t>::set_loss(real_t loss,
    const real_t* Y, const real_t* loss_weights)
{
    if (loss < ZERO || loss > ONE){
        cerr << "PFDR graph d1 loss simplex: loss parameter should be between "
            "0 and 1 (" << loss << " given)." << endl;
        exit(EXIT_FAILURE);
    }
    if ((this->loss != loss) &&
        (this->loss == LINEAR || this->loss == QUADRATIC ||
         loss == LINEAR || loss == QUADRATIC)){
        cerr << "PFDR graph d1 loss simplex: the type of loss cannot "
            "be changed; for changing from one loss type to another, create "
            "a new instance of Pfdr_d1_lsx." << endl;
        exit(EXIT_FAILURE);
    }
    this->loss = loss;
    if (Y){ this->Y = Y; }
    if (loss_weights && loss == LINEAR){
        cerr << "PFDR d1 loss simplex: with linear loss, weights "
            "should be directly incorporated in the observations Y." << endl;
        exit(EXIT_FAILURE);
    }
    this->loss_weights = loss_weights;
}

template<typename real_t, typename vertex_t>
void Pfdr_d1_lsx<real_t, vertex_t>::compute_lipschitz_metric()
{
    if (loss == LINEAR){
        l = ZERO; lshape = SCALAR;
    }else if (loss == QUADRATIC){
        if (loss_weights){ L = loss_weights; lshape = MONODIM; }
        else{ l = ONE; lshape = SCALAR; }
    }else{ /* KLs loss, Ld = max_{0 <= x_d <= 1} d^2KLs/dx_d^2
            *              = (1-s)^2/(s/D)^2 (s/D + (1-s)y_d) */
        real_t c = (ONE - loss);
        real_t q = loss/D;
        real_t r = c*c/(q*q);
        Lmut = (real_t*) malloc_check(sizeof(real_t)*V*D); 
        #pragma omp parallel for schedule(static) NUM_THREADS(2*V*D, V)
        for (vertex_t v = 0; v < V; v++){
            size_t vd = v*D;
            for (size_t d = 0; d < D; d++){
                Lmut[vd] = LOSS_WEIGHTS_(v)*r*(q + c*Y[vd]);
                vd++;
            }
        }
        L = Lmut; lshape = MULTIDIM;
    }
}

template<typename real_t, typename vertex_t>
void Pfdr_d1_lsx<real_t, vertex_t>::compute_hess_f()
{
    const size_t Dga = gashape == MULTIDIM ? D : 1;
    if (loss == LINEAR){
        for (size_t vd = 0; vd < V*Dga; vd++){ Ga[vd] = ZERO; }
    }else if (loss == QUADRATIC){
        for (vertex_t v = 0; v < V; v++){
            size_t vd = v*Dga;
            for (size_t d = 0; d < Dga; d++){ Ga[vd++] = LOSS_WEIGHTS_(v); }
        }
    }else{ /* d^2KLs/dx_d^2 = (1-s)^2 (s/D + (1-s)y_d)/(s/D + (1-s)x_d)^2 */
        real_t c = (ONE - loss);
        real_t q = loss/D;
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V)
        for (vertex_t v = 0; v < V; v++){
            size_t vd = v*D;
            for (size_t d = 0; d < D; d++){
                real_t r = c/(q + c*X[vd]);
                Ga[vd] = LOSS_WEIGHTS_(v)*(q + c*Y[vd])*r*r;
                vd++;
            }
        }
    }
}

template <typename real_t, typename vertex_t>
void Pfdr_d1_lsx<real_t, vertex_t>::compute_Ga_grad_f()
{
    /**  forward and backward steps on auxiliary variables  **/
    /* explicit step */
    if (loss == LINEAR){ /* linear loss, grad = -Y */
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V)
        for (vertex_t v = 0; v < V; v++){
            size_t vd = v*D;
            for (size_t d = 0; d < D; d++){
                Ga_grad_f[vd] = -Ga_(v, vd)*Y[vd];
                vd++;
             }
        }
    }else if (loss == QUADRATIC){ /* quadratic loss, grad = w (X - Y) */
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V)
        for (vertex_t v = 0; v < V; v++){
            size_t vd = v*D;
            for (size_t d = 0; d < D; d++){
                Ga_grad_f[vd] = LOSS_WEIGHTS_(v)*Ga_(v, vd)*(X[vd] - Y[vd]);
                vd++;
            }
        }
    }else{ /* dKLs/dx_k = -(1-s)(s/D + (1-s)y_k)/(s/D + (1-s)x_k) */
        real_t r = loss/D/(ONE - loss);
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D)
        for (size_t vd = 0; vd < V*D; vd++){
            Ga_grad_f[vd] = KL_Ga_Y[vd]/(r + X[vd]);
        }
    }
}

template<typename real_t, typename vertex_t>
void Pfdr_d1_lsx<real_t, vertex_t>::compute_prox_Ga_h()
{
    if (gashape == MULTIDIM){
        proj_simplex<real_t>(X, D, V, nullptr, ONE, Ga);
    }else{
        proj_simplex<real_t>(X, D, V, nullptr, ONE);
    }
}

template<typename real_t, typename vertex_t>
real_t Pfdr_d1_lsx<real_t, vertex_t>::compute_f()
{
    real_t obj = ZERO;
    if (loss == LINEAR){
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D) \
            reduction(+:obj)
        for (size_t vd = 0; vd < V*D; vd++){ obj -= X[vd]*Y[vd]; }
    }else if (loss == QUADRATIC){
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
            reduction(+:obj)
        for (vertex_t v = 0; v < V; v++){
            size_t vd = v*D;
            real_t dif2 = ZERO;
            for (size_t d = 0; d < D; d++){
                dif2 += (X[vd] - Y[vd])*(X[vd] - Y[vd]);
                vd++;
            }
            obj += LOSS_WEIGHTS_(v)*dif2;
        }
        obj *= HALF;
    }else{ /* smoothed Kullback-Leibler */
        real_t c = (ONE - loss);
        real_t q = loss/D;
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
            reduction(+:obj) 
        for (vertex_t v = 0; v < V; v++){
            size_t vd = v*D;
            real_t KLs = ZERO;
            for (size_t d = 0; d < D; d++){
                real_t ys = q + c*Y[vd];
                KLs += ys*log(ys/(q + c*X[vd]));
                vd++;
            }
            obj += LOSS_WEIGHTS_(v)*KLs;
        }
    }
    return obj;
}

template<typename real_t, typename vertex_t>
void Pfdr_d1_lsx<real_t, vertex_t>::preconditioning(bool init)
{
    Pfdr_d1<real_t, vertex_t>::preconditioning(init);

    /* precompute first-order information for Kullback-Leibler loss gradient */
    if (loss != LINEAR && loss != QUADRATIC){
        if (!KL_Ga_Y){ KL_Ga_Y = (real_t*) malloc_check(sizeof(real_t)*V*D); }
        /* dKLs/dx_d = -(1-s)(s/D + (1-s)y_d)/(s/K + (1-s)x_d) */
        real_t c = (ONE - loss);
        real_t q = loss/D;
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V)
        for (vertex_t v = 0; v < V; v++){
            size_t vd = v*D;
            for (size_t d = 0; d < D; d++){
                KL_Ga_Y[vd] = -LOSS_WEIGHTS_(v)*Ga[vd]*(q + c*Y[vd]);
                vd++;
            }
        }
    }
}

template<typename real_t, typename vertex_t>
void Pfdr_d1_lsx<real_t, vertex_t>::initialize_iterate()
{
    if (loss == LINEAR){ /* Yv might not lie on the simplex;
        * create a point on the simplex by removing the minimum value
        * (resulting problem loss + d1 + simplex problem strictly equivalent)
        * and dividing by the sum */
        #pragma omp parallel for schedule(static) NUM_THREADS(2*V*D, V)
        for (vertex_t v = 0; v < V; v++){
            const real_t* Yv = Y + v*D;
            real_t* Xv = X + v*D;
            real_t min = Yv[0], max = Yv[0], sum = Yv[0];
            for (size_t d = 1; d < D; d++){
                sum += Yv[d];
                if (Yv[d] < min){ min = Yv[d]; }
                else if (Yv[d] > max){ max = Yv[d]; }
            }
            if (min == max){ // avoid trouble if all equal
                for (size_t d = 0; d < D; d++){ Xv[d] = ONE/D; }
            }else{
                sum -= D*min;
                for (size_t d = 0; d < D; d++){
                    Xv[d] = (Yv[d] - min)/sum;
                }
            }
        }
    }else{ /* Yv lies on the simplex */
        for (size_t vd = 0; vd < V*D; vd++){ X[vd] = Y[vd]; }
    }
}

template <typename real_t, typename vertex_t>
real_t Pfdr_d1_lsx<real_t, vertex_t>::compute_evolution()
{
    real_t dif = ZERO;
    #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
        reduction(+:dif)
    for (vertex_t v = 0; v < V; v++){
        size_t vd = v*D;
        for (size_t d = 0; d < D; d++){
            dif += abs(last_X[vd] - X[vd]);
            last_X[vd] = X[vd];
            vd++;
        }
    }
    return dif/V;
}

/**  instantiate for compilation  **/
template class Pfdr_d1_lsx<float, uint16_t>;

template class Pfdr_d1_lsx<float, uint32_t>;

template class Pfdr_d1_lsx<double, uint16_t>;

template class Pfdr_d1_lsx<double, uint32_t>;
