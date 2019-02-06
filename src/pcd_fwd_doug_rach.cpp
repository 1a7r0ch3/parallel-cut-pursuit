/*=============================================================================
 * Hugo Raguet 2016
 *===========================================================================*/
#include "../include/omp_num_threads.hpp"
#include "../include/pcd_fwd_doug_rach.hpp"

/* constants of the correct type */
#define ZERO ((real_t) 0.)
#define ONE ((real_t) 1.)
#define TWO ((real_t) 2.)
#define ALMOST_TWO ((real_t) 1.9)

/* macros for indexing conditioners depending on their shape */
#define W_(j, jd)    (wshape == SCALAR ? W[(j)/size] : \
                      wshape == MONODIM ? W[(j)] : W[(jd)])
#define Ga_(i, id)   (gashape == SCALAR ? ga : \
                      gashape == MONODIM ? Ga[(i)] : Ga[(id)])
#define L_(i, id)    (lshape == SCALAR ? l : \
                      lshape == MONODIM ? L[(i)] : L[(id)])
#define Id_W_(i, id) (wshape == MONODIM ? Id_W[(i)] : Id_W[(id)])
#define aux_idx_(j) (aux_idx ? aux_idx[(j)] : ((j) % size))

using namespace std;

template <typename real_t, typename index_t>
Pfdr<real_t, index_t>::Pfdr(index_t size, size_t aux_size,
    const index_t* aux_idx, size_t D, Condshape gashape, Condshape wshape) :
    Pcd_prox<real_t>(size*D), size(size), aux_size(aux_size),
    D(D), aux_idx(aux_idx), gashape(gashape), wshape(wshape)
{
    set_name("Preconditioned forward-Douglas-Rachford algorithm");
    rho = ONE;
    L = Lmut = nullptr;
    l = ZERO; lshape = SCALAR;
    lipschcomput = EACH;
    Ga = Ga_grad_f = Z = W = Z_Id = Id_W = nullptr;
}

template <typename real_t, typename index_t>
Pfdr<real_t, index_t>::~Pfdr()
{ free(Ga); free(Z); free(W); free(Ga_grad_f); free(Lmut); }

template <typename real_t, typename index_t>
void Pfdr<real_t, index_t>::set_relaxation(real_t rho){ this->rho = rho; }

template <typename real_t, typename index_t>
void Pfdr<real_t, index_t>::set_lipschitz_param(const real_t* L,
    real_t l, Condshape lshape)
{
    this->L = L;
    this->l = l;
    if (L){ this->lshape = lshape < gashape ? lshape : gashape; }
    else{ this->lshape = SCALAR; }
    this->lipschcomput = USER;
}

template <typename real_t, typename index_t>
void Pfdr<real_t, index_t>::set_lipschitz_param(Lipschcomput lipschcomput)
{ this->lipschcomput = lipschcomput; }

template <typename real_t, typename index_t>
void Pfdr<real_t, index_t>::set_auxiliary(real_t* Z){ this->Z = Z; }

template <typename real_t, typename index_t>
real_t* Pfdr<real_t, index_t>::get_auxiliary(){ return this->Z; }

template <typename real_t, typename index_t>
void Pfdr<real_t, index_t>::compute_lipschitz_metric()
{ l = ZERO; lshape = SCALAR; }

template <typename real_t, typename index_t>
void Pfdr<real_t, index_t>::compute_hess_f()
/* default to zero f, can be overriden */
{
    for (index_t i = 0; i < size; i++){
        size_t id = i*D;
        for (size_t d = 0; d < D; d++){ Ga_(i, id++) = ZERO; }
    }
}

template <typename real_t, typename index_t>
void Pfdr<real_t, index_t>::add_pseudo_hess_h()
/* default to zero h, can be overriden */ {}

template <typename real_t, typename index_t>
void Pfdr<real_t, index_t>::make_sum_Wi_Id()
{
    if (wshape == SCALAR){
        size_t n = aux_size/size;
        real_t sum_wi = ZERO;
        for (index_t i = 0; i < n; i++){ sum_wi += W[i]; }
        for (index_t i = 0; i < n; i++){ W[i] /= sum_wi; }
    }else{
        if (!Id_W){ /* normalize by the sum */
            const size_t Dw = wshape == MULTIDIM ? D : 1;
            /* compute sum */
            real_t* sum_Wi = (real_t*) malloc_check(sizeof(real_t)*size*Dw);
            for (size_t id = 0; id < size*Dw; id++){ sum_Wi[id] = ZERO; }
            #pragma omp parallel for schedule(static) \
                NUM_THREADS(aux_size*Dw, Dw)
            for (size_t d = 0; d < Dw; d++){
                size_t jd = d;
                for (size_t j = 0; j < aux_size; j++){
                    sum_Wi[d + aux_idx_(j)*Dw] += W[jd];
                    jd += Dw;
                }
            }
            /* normalize */
            #pragma omp parallel for schedule(static) \
                NUM_THREADS(aux_size*Dw, aux_size)
            for (size_t j = 0; j < aux_size; j++){
                size_t id = aux_idx_(j)*Dw;
                size_t jd = j*Dw;
                for (size_t d = 0; d < D; d++){ W[jd++] /= sum_Wi[id++]; }
            }
            free(sum_Wi);
        }else{ /* metric must be kept */            
            cerr << "PFDR: a specialization of the virtual "
                "function make_sum_Wi_Id() must be provided in order to use "
                "the weights Wi to shape the metric of the proximity operators"
                " of g." << endl;
            exit(EXIT_FAILURE);
        }
    }
}

template <typename real_t, typename index_t>
void Pfdr<real_t, index_t>::initialize_auxiliary(){
    if (!Z){ Z = (real_t*) malloc_check(sizeof(real_t)*aux_size*D); }
    for (size_t j = 0; j < aux_size; j++){
        size_t id = aux_idx_(j)*D;
        size_t jd = j*D;
        for (size_t d = 0; d < D; d++){ Z[jd++] = X[id++]; }
    }
    if (Z_Id){ for (size_t id = 0; id < size*D; id++){ Z_Id[id] = X[id]; } }
}

template<typename real_t, typename index_t>
void Pfdr<real_t, index_t>::compute_Ga_grad_f()
/* default to zero f, can be overriden */
{ for (size_t id = 0; id < size*D; id++){ Ga_grad_f[id] = ZERO; } }

template <typename real_t, typename index_t>
void Pfdr<real_t, index_t>::compute_weighted_average()
{
    #pragma omp parallel for schedule(static) NUM_THREADS(aux_size*D, D)
    for (size_t d = 0; d < D; d++){ 
        size_t id = d;
        for (index_t i = 0; i < size; i++){
            X[id] = Id_W ?
                (Id_W_(i, id)*(Z_Id ? Z_Id[id] : Ga_grad_f[id] - X[id])) : ZERO;
            id += D;
        }
        size_t jd = d;
        for (size_t j = 0; j < aux_size; j++){
            X[d+aux_idx_(j)*D] += W_(j, jd)*Z[jd];
            jd += D;
        }
    }
}

template<typename real_t, typename index_t>
void Pfdr<real_t, index_t>::compute_prox_Ga_h()
/* default to zero f, can be overriden */ {}

template<typename real_t, typename index_t>
real_t Pfdr<real_t, index_t>::compute_f()
/* default to zero f, can be overriden */ { return ZERO; }

template<typename real_t, typename index_t>
real_t Pfdr<real_t, index_t>::compute_h()
/* default to zero f, can be overriden */ { return ZERO; }

template<typename real_t, typename index_t>
void Pfdr<real_t, index_t>::preconditioning(bool init)
{
    Pcd_prox<real_t>::preconditioning(init);

    if (init){
        if (!Z){ initialize_auxiliary(); }
        if (!Ga && gashape != SCALAR){
            if (gashape == MONODIM){
                Ga = (real_t*) malloc_check(sizeof(real_t)*size);
            }else{
                Ga = (real_t*) malloc_check(sizeof(real_t)*size*D);
            }
        }
        if (!W){
            if (wshape == SCALAR){
                W = (real_t*) malloc_check(sizeof(real_t)*aux_size/size);
            }else if (wshape == MONODIM){
                W = (real_t*) malloc_check(sizeof(real_t)*aux_size);
            }else{
                W = (real_t*) malloc_check(sizeof(real_t)*aux_size*D);
            }
        }
        if (!Ga_grad_f){
            Ga_grad_f = (real_t*) malloc_check(sizeof(real_t)*size*D);
        }
    }else{ /**  compute the auxiliary subgradients in Z  **/
        compute_Ga_grad_f(); 
        #pragma omp parallel for schedule(static) \
            NUM_THREADS(D*4*aux_size, aux_size)
        for (size_t j = 0; j < aux_size; j++){
            index_t i = aux_idx_(j);
            size_t id = i*D;
            size_t jd = j*D;
            for (size_t d = 0; d < D; d++){
                Z[jd] = (W_(j, jd)/Ga_(i, id))*(X[id] - Ga_grad_f[id] - Z[jd]);
                id++; jd++;
            }
        }
        if (Z_Id){
            #pragma omp parallel for schedule(static) NUM_THREADS(D*size, size)
            for (index_t i = 0; i < size; i++){
                size_t id = id*D;
                for (size_t d = 0; d < D; d++){
                    Z_Id[id] = (Id_W_(i, id)/Ga_(i, id))
                        *(X[id] - Ga_grad_f[id] - Z_Id[id]);
                    id++;
                }
            }
        }
    }

    /**  second-order information on f  **/
    if (lipschcomput == EACH || (lipschcomput == ONCE && !L && !l)){
        compute_lipschitz_metric();
    }

    compute_hess_f();

    add_pseudo_hess_g();

    add_pseudo_hess_h();

    const size_t Dga = gashape == MULTIDIM ? D : 1;
    const index_t sizega = gashape == SCALAR ? 1 : size;

    /**  inverse the pseudo-Hessian  **/
    #pragma omp parallel for schedule(static) NUM_THREADS(sizega*Dga)
    for (size_t id = 0; id < sizega*Dga; id++){ Ga_(id, id) = ONE/Ga_(id, id); }

    /**  convergence condition on the metric and stability  **/ 
    real_t lga_max = ALMOST_TWO*(TWO - rho);
    #pragma omp parallel for schedule(static) NUM_THREADS(sizega*Dga, sizega)
    for (index_t i = 0; i < sizega; i++){
        size_t id = i*Dga;
        for (size_t d = 0; d < Dga; d++){
            real_t ga_max = lga_max/L_(i, id);
            real_t ga_min = L_(i, id) > ZERO ? cond_min*ga_max : cond_min;
            if (Ga_(i, id) > ga_max){ Ga_(i, id) = ga_max; }
            else if (Ga_(i, id) < ga_min){ Ga_(i, id) = ga_min; }
            id++;
        }
    }

    if (lipschcomput == EACH){ free(Lmut); L = Lmut = nullptr; }

    make_sum_Wi_Id();

    if (!init){ /**  update auxiliary variables  **/
        #pragma omp parallel for schedule(static) \
            NUM_THREADS(2*aux_size*D, aux_size)
        for (size_t j = 0; j < aux_size; j++){
            index_t i = aux_idx_(j);
            size_t id = i*D;
            size_t jd = j*D;
            for (size_t d = 0; d < D; d++){
                Z[jd] = X[id] - Ga_grad_f[id] - Ga_(i, id)*Z[jd]/W_(j, jd);
                id++; jd++;
            }
        }
        if (Z_Id){
            #pragma omp parallel for schedule(static) NUM_THREADS(D*size, size)
            for (index_t i = 0; i < size; i++){
                size_t id = i*D;
                for (size_t d = 0; d < D; d++){
                    Z_Id[id] = X[id] - Ga_grad_f[id] -
                        Ga_(i, id)*Z_Id[id]/Id_W_(i, id);
                    id++;
                }
            }
        }
    }
}

template <typename real_t, typename index_t>
void Pfdr<real_t, index_t>::main_iteration()
{
    /* gradient step, forward = 2 X - Zi - Ga grad(X) */ 
    compute_Ga_grad_f();
    #pragma omp parallel for schedule(static) NUM_THREADS(size*D)
    for (size_t id = 0; id < size*D; id++){
        Ga_grad_f[id] = TWO*X[id] - Ga_grad_f[id];
    }

    /* generalized forward-backward step on auxiliary Z */
    compute_prox_GaW_g(); 
    if (Z_Id){ /* take care of the additional auxiliary variable */
        for (size_t id = 0; id < size*D; id++){
            Z_Id[id] += rho*(Ga_grad_f[id] - Z_Id[id] - X[id]); id++;
        }
    }

    /* projection on first diagonal */
    compute_weighted_average();

    /* backward step on iterate X */
    compute_prox_Ga_h(); 
}

template <typename real_t, typename index_t>
real_t Pfdr<real_t, index_t>::compute_objective()
{ return compute_f() + compute_g() + compute_h(); }

/**  instantiate for compilation  **/
template class Pfdr<float, uint16_t>;

template class Pfdr<float, uint32_t>;

template class Pfdr<double, uint16_t>;

template class Pfdr<double, uint32_t>;
