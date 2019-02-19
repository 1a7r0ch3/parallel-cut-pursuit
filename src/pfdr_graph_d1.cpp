/*=============================================================================
 * Hugo Raguet 2016
 *===========================================================================*/
#include <cmath>
#include "../include/omp_num_threads.hpp"
#include "../include/pfdr_graph_d1.hpp"

/* constants of the correct type */
#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)
#define TWO ((real_t) 2.0)
#define HALF ((real_t) 0.5)

/* macros for indexing data arrays depending on their shape */
#define EDGE_WEIGHTS_(e) (edge_weights ? edge_weights[(e)] : homo_edge_weight)
#define COOR_WEIGHTS_(d) (coor_weights ? coor_weights[(d)] : ONE)
#define W_d1_(i, id)  (wd1shape == SCALAR ? w_d1 : \
                       wd1shape == MONODIM ? W_d1[(i)] : W_d1[(id)])
#define Th_d1_(e, ed) (thd1shape == SCALAR ? th_d1 : \
                       thd1shape == MONODIM ? Th_d1[(e)] : Th_d1[(ed)])

#define TPL template <typename real_t, typename vertex_t>
#define PFDR_D1 Pfdr_d1<real_t, vertex_t>

using namespace std;

TPL PFDR_D1::Pfdr_d1(vertex_t V, size_t E, const vertex_t* edges, size_t D,
    D1p d1p, const real_t* coor_weights, Condshape hess_f_h_shape) :
    Pfdr<real_t, vertex_t>(V, 2*E, edges, D,
        compute_ga_shape(coor_weights, hess_f_h_shape),
        compute_w_shape(d1p, coor_weights, hess_f_h_shape)),
    E(E), d1p(d1p), coor_weights(coor_weights),
    wd1shape(compute_wd1_shape(d1p, coor_weights, hess_f_h_shape)),
    thd1shape(compute_thd1_shape(d1p, coor_weights, hess_f_h_shape))
{
    edge_weights = nullptr;
    homo_edge_weight = ONE;
    W_d1 = Th_d1 = nullptr;
}

TPL PFDR_D1::~Pfdr_d1(){ free(W_d1); free(Th_d1); }

TPL void PFDR_D1::set_edge_weights(const real_t* edge_weights,
    real_t homo_edge_weight, const real_t* coor_weights)
{
    this->edge_weights = edge_weights;
    this->homo_edge_weight = homo_edge_weight;
    if (!this->coor_weights != !coor_weights){
        cerr << "PFDR graph d1: coor_weights attribute cannot be "
            "changed from null to varying weights or vice versa; for changing "
            "these weights, create a new instance of Pfdr_d1." << endl;
        exit(EXIT_FAILURE);
    }
    this->coor_weights = coor_weights;
}

TPL void PFDR_D1::add_pseudo_hess_g()
/* d1 contribution and splitting weights
 * a local quadratic approximation of (x1,x2) -> ||x1 - x2|| at (y1,y2) is
 * x -> 1/2 (||x1 - x2||^2/||y1 - y2|| + ||y1 - y2||)
 * whose hessian is not diagonal, but keeping only the diagonal terms yields
 * the second order derivative 1/||y1 - y2|| */
{
    /* finite differences and amplitudes */
    #pragma omp parallel for schedule(static) NUM_THREADS(4*E, E)
    for (size_t e = 0; e < E; e++){
        real_t* Xu = X + edges[2*e]*D;
        real_t* Xv = X + edges[2*e + 1]*D;
        real_t dif = ZERO, ampu = ZERO, ampv = ZERO;
        for (size_t d = 0; d < D; d++){
            if (d1p == D11){
                dif += abs(Xu[d] - Xv[d])*COOR_WEIGHTS_(d);
                ampu += abs(Xu[d])*COOR_WEIGHTS_(d);
                ampv += abs(Xv[d])*COOR_WEIGHTS_(d);
            }else{
                dif += (Xu[d] - Xv[d])*(Xu[d] - Xv[d])*COOR_WEIGHTS_(d);
                ampu += Xu[d]*Xu[d]*COOR_WEIGHTS_(d);
                ampv += Xv[d]*Xv[d]*COOR_WEIGHTS_(d);
            }
        }
        real_t amp;
        if (d1p == D11){
            amp = ampu > ampv ? ampu : ampv;
        }else{
            dif = sqrt(dif);
            amp = ampu > ampv ? sqrt(ampu) : sqrt(ampv);
        }
        /* stability of the preconditioning */
        if (dif < amp*cond_min){ dif = amp*cond_min; }
        if (dif < eps){ dif = eps; }
        Th_d1[e] = EDGE_WEIGHTS_(e)/dif; /* use Th_d1 as temporary storage */
    }

    /* actual pseudo-hessian, can be parallelized along coordinates */
    const size_t Dga = gashape == MULTIDIM ? D : 1;
    const size_t Dw = wshape == MULTIDIM ? D : 1; /* Dw <= Dga */
    #pragma omp parallel for schedule(static) NUM_THREADS(4*E*Dga, Dga)
    for (size_t d = 0; d < Dga; d++){
        size_t id = d;
        size_t jd = d + Dw;
        for (size_t e = 0; e < E; e++){
            real_t coef = COOR_WEIGHTS_(d)*Th_d1[e];
            Ga[d + edges[2*e]*Dga] += coef;
            Ga[d + edges[2*e + 1]*Dga] += coef;
            if (!Id_W && (wshape == MULTIDIM || d == 0)){
                W[id] = W[jd] = coef;
                id += 2*Dw;
                jd += 2*Dw;
            }
        }
    }
}

TPL void PFDR_D1::make_sum_Wi_Id()
{
    /* compute splitting weights sum */
    real_t* sum_Wi;
    /* use temporary storage if available */
    const size_t Dwd1 = wd1shape == MULTIDIM ? D : wd1shape == MONODIM ? 1 : 0; 
    const size_t Dthd1 = thd1shape == MULTIDIM ? D : 1; 

    if (2*E*Dwd1 >= V){ sum_Wi = W_d1; }
    else if (E*Dthd1 >= V){ sum_Wi = Th_d1; }
    else{ sum_Wi = (real_t*) malloc_check(sizeof(real_t)*V); }

    for (size_t v = 0; v < V; v++){ sum_Wi[v] = ZERO; }
    for (size_t e = 0; e < 2*E; e++){ sum_Wi[edges[e]] += Id_W ? ONE : W[e]; }

    if (!Id_W){ /* weights can just be normalized */

        #pragma omp parallel for schedule(static) NUM_THREADS(2*E)
        for (size_t e = 0; e < 2*E; e++){ W[e] /= sum_Wi[edges[e]]; }

    }else{ /* weights are used in order to shape the metric */
        /* compute shape and maximum */
        #pragma omp parallel for schedule(static) NUM_THREADS(2*V*D, V)
        for (size_t v = 0; v < V; v++){
            size_t vd = v*D;
            real_t wmax = Id_W[vd] = Ga[vd]*COOR_WEIGHTS_(0);
            vd++;
            for (size_t d = 1; d < D; d++){
                Id_W[vd] = Ga[vd]*COOR_WEIGHTS_(d);
                if (Id_W[vd] > wmax){ wmax = Id_W[vd]; }
                vd++;
            }
            vd -= D;
            for (size_t d = 0; d < D; d++){
                Id_W[vd] = ONE - Id_W[vd]/wmax;
                vd++;
            }
        }
        /* set weights */
        #pragma omp parallel for schedule(static) NUM_THREADS(2*E*D, 2*E)
        for (size_t e = 0; e < 2*E; e++){
            size_t v = edges[e];
            size_t vd = v*D;
            size_t ed = e*D;
            for (size_t d = 0; d < D; d++){
                W[ed++] = (ONE - Id_W[vd++])/sum_Wi[v];
            }
        }
    }

    if (2*E*Dwd1 < V && E*Dthd1 < V){ free(sum_Wi); }
}

TPL void PFDR_D1::preconditioning(bool init)
{
    /* allocate weights and thresholds for d1 prox operator */
    if (!W_d1 && wd1shape != SCALAR){
        size_t wd1size = 2*E*(wd1shape == MULTIDIM ? D : 1);
        W_d1 = (real_t*) malloc_check(sizeof(real_t)*wd1size);
    }
    if (!Th_d1){
        size_t thd1size = E*(thd1shape == MULTIDIM ? D : 1);
        Th_d1 = (real_t*) malloc_check(sizeof(real_t)*thd1size);
    }

    /* allocate supplementary weights and auxiliary variables if necessary */
    if (!Id_W && wshape == MULTIDIM){
        Id_W = (real_t*) malloc_check(sizeof(real_t)*V*D);
        if (!Z_Id && Pfdr<real_t, vertex_t>::rho != ONE){
            Z_Id = (real_t*) malloc_check(sizeof(real_t)*V*D);
        }
    }

    Pfdr<real_t, vertex_t>::preconditioning(init);

    /* precompute weights and thresholds for d1 prox operator */
    if (wd1shape == SCALAR){ w_d1 = HALF; }
    const size_t Dd1 = thd1shape == MULTIDIM ? D : 1;
    const size_t Dga = gashape == MULTIDIM ? D : 1;
    const size_t Dw = wshape == MULTIDIM ? D : 1;
    #pragma omp parallel for schedule(static) NUM_THREADS(8*E*Dd1, E)
    for (size_t e = 0; e < E; e++){
        size_t i = 2*e;
        size_t j = 2*e + 1;
        vertex_t u = edges[i];
        vertex_t v = edges[j];
        size_t ud = u*Dga;
        size_t vd = v*Dga;
        size_t ed = e*Dd1;
        size_t id, jd;
        if (wd1shape != SCALAR){
            id = i*Dd1;
            jd = j*Dd1;
        }
        for (size_t d = 0; d < Dd1; d++){
            real_t w_ga_u = W[i*Dw]/Ga[ud++];
            real_t w_ga_v = W[j*Dw]/Ga[vd++];
            Th_d1[ed++] = EDGE_WEIGHTS_(e)*COOR_WEIGHTS_(d)
                *(w_ga_u + w_ga_v)/(w_ga_u*w_ga_v);
            if (wd1shape != SCALAR){
                W_d1[id++] = w_ga_u/(w_ga_u + w_ga_v);
                W_d1[jd++] = w_ga_v/(w_ga_u + w_ga_v);
            }
        }
    }
}

TPL void PFDR_D1::compute_prox_GaW_g()
{
    #pragma omp parallel for schedule(static) NUM_THREADS(8*E*D, E)
    for (size_t e = 0; e < E; e++){
        size_t i = 2*e;
        size_t j = 2*e + 1;
        size_t ud = edges[i]*D;
        size_t vd = edges[j]*D;
        size_t id = i*D;
        size_t jd = j*D;
        size_t ed;
        real_t thresholding, dnorm;
        if (d1p == D12){ /* compute norm and threshold */
            dnorm = ZERO;
            for (size_t d = 0; d < D; d++){ 
                /* forward step */ 
                real_t fwd_zi = Ga_grad_f[ud++] - Z[id++];
                real_t fwd_zj = Ga_grad_f[vd++] - Z[jd++];
                dnorm += (fwd_zi - fwd_zj)*(fwd_zi - fwd_zj)*COOR_WEIGHTS_(d);
            }
            dnorm = sqrt(dnorm);
            thresholding = dnorm > Th_d1[e] ? ONE - Th_d1[e]/dnorm : ZERO;
            ud -= D; vd -= D; id -= D; jd -= D;
        }else if (thd1shape == MULTIDIM){
            ed = e*D;
        }
        /* soft thresholding, update and relaxation */
        for (size_t d = 0; d < D; d++){
            /* forward step */ 
            real_t fwd_zi = Ga_grad_f[ud] - Z[id];
            real_t fwd_zj = Ga_grad_f[vd] - Z[jd];
            /* backward step */
            real_t avg = W_d1_(i, id)*fwd_zi + W_d1_(j, jd)*fwd_zj;
            real_t dif = fwd_zi - fwd_zj;
            if (d1p == D11){
                if (dif > Th_d1_(e, ed)){ dif -= Th_d1_(e, ed); }
                else if (dif < -Th_d1_(e, ed)){ dif += Th_d1_(e, ed); }
                else{ dif = ZERO; }
                if (thd1shape == MULTIDIM){ ed++; }
            }else{
                dif *= thresholding;
            }
            Z[id] += rho*(avg + W_d1_(j, jd)*dif - X[ud]);
            Z[jd] += rho*(avg - W_d1_(i, id)*dif - X[vd]);
            ud++; vd++; id++; jd++;
        }
    }
}

TPL real_t PFDR_D1::compute_g()
{
    real_t obj = ZERO;
    #pragma omp parallel for schedule(static) NUM_THREADS(2*E*D, E) \
        reduction(+:obj)
    for (size_t e = 0; e < E; e++){
        size_t ud = edges[2*e]*D;
        size_t vd = edges[2*e + 1]*D;
        real_t dif = ZERO;
        for (size_t d = 0; d < D; d++){
            if (d1p == D11){
                dif += abs(X[ud] - X[vd])*COOR_WEIGHTS_(d);
            }else{
                dif += (X[ud] - X[vd])*(X[ud] - X[vd])*COOR_WEIGHTS_(d);
            }
            ud++; vd++;
        }
        if (d1p == D12){ dif = sqrt(dif); }
        obj += EDGE_WEIGHTS_(e)*dif;
    }
    return obj;
}

/* instantiate for compilation */
template class Pfdr_d1<float, uint16_t>;
template class Pfdr_d1<float, uint32_t>;
template class Pfdr_d1<double, uint16_t>;
template class Pfdr_d1<double, uint32_t>;
