/*=============================================================================
 * Minimize functional over a graph G = (V, E)
 *
 *        F(x) = f(x) + ||x||_d1 + i_{simplex}(x)
 *
 * where for each vertex, x_v is a D-dimensional vector,
 *       f is a separable data-fidelity loss
 *       ||x||_d1 = sum_{uv in E} w_d1_uv (sum_d w_d1_d |x_ud - x_vd|),
 * and i_{simplex} is the standard D-simplex constraint over each vertex,
 *     i_{simplex} = 0 for all v, (for all d, x_vd >= 0) and sum_d x_vd = 1,
 *                 = infinity otherwise;
 *
 * using preconditioned forward-Douglas-Rachford splitting algorithm.
 *
 * Parallel implementation with OpenMP API.
 *
 * H. Raguet and L. Landrieu, Preconditioning of a Generalized Forward-Backward
 * Splitting and Application to Optimization on Graphs, SIAM Journal on Imaging
 * Sciences, 2015, 8, 2706-2739
 *
 * H. Raguet, A Note on the Forward-Douglas-Rachford Splitting for Monotone 
 * Inclusion and Convex Optimization, Optimization Letters, 2018, 1-24
 *
 * Hugo Raguet 2016, 2018
 *===========================================================================*/
#pragma once
#include "pfdr_graph_d1.hpp"
#define LINEAR ((real_t) 0.0)
#define QUADRATIC ((real_t) 1.0)

/* vertex_t is an integer type able to represent the number of vertices */
template <typename real_t, typename vertex_t>
class Pfdr_d1_lsx : public Pfdr_d1<real_t, vertex_t>
{
public:
    /**  constructor, destructor  **/

    Pfdr_d1_lsx(vertex_t V, size_t E, const vertex_t* edges, real_t loss,
        size_t D, const real_t* Y, const real_t* d1_coor_weights = nullptr);

    /* the destructor does not free pointers which are supposed to be provided 
     * by the user (adjacency graph structure given at construction, 
     * monitoring arrays, matrix and observation arrays); it does free the rest 
     * (iterate, auxiliary variables etc.), but this can be prevented by
     * copying the corresponding pointer member and set it to null before
     * deleting */
	~Pfdr_d1_lsx();

    /**  methods for manipulating parameters  **/
 
    void initialize_iterate() override; // initialize on simplex based on Y

    /* warning: the first parameter loss can only be used to change the 
     * smoothing parameter of a Kullback-Leibler loss; for changing from one
     * loss type to another, create a new instance; Y is changed only if the
     * argument is not null */
    void set_loss(real_t loss, const real_t* Y = nullptr,
        const real_t* loss_weights = nullptr);

    /* overload for changing only loss_weights, or the observations Y when the
     * loss is linear (in which case loss_weights is ignored) */
    void set_loss(const real_t* loss_weights_or_Y)
    {
        if (loss == LINEAR){ set_loss(loss, loss_weights_or_Y); }
        else{ set_loss(loss, nullptr, loss_weights_or_Y); }
    }

private:
    /**  separable loss term  **/

    /* observations, D-by-V array, column major format;
     * for losses other than linear, they are supposed to lie on the simplex */
    const real_t* Y; 

    /* 0 for linear (macro LINEAR)
     *     f(x) = - <x, y> ,  with  <x, y> = sum_{v,d} x_{v,d} y_{v,d} ;
     * 1 for quadratic (macro QUADRATIC)
     *     f(x) = 1/2 ||y - x||_{l2,w}^2 ,
     * with  ||y - x||_{l2,w}^2 = sum_{v,d} w_v (y_{v,d} - x_{v,d})^2 ;
     * 0 < loss < 1 for smoothed Kullback-Leibler divergence
     *     f(x) = sum_v w_v KLs(x_v, y_v),
     * with KLs(y_v, x_v) = KL(s u + (1 - s) y_v ,  s u + (1 - s) x_v),
     * where KL is the regular Kullback-Leibler divergence,
     *       u is the uniform discrete distribution over {1,...,D}, and
     *       s = loss is the smoothing parameter ;
     * It yields, 
     *     KLs(y_v, x_v) = - sum_k (s/D + (1 - s) y_v) log(s/D + (1 - s) x_v)
     *                     - H(s u + (1 - s) y_v) ,
     * where the constant - H(s u + (1 - s) y_v) is equal to
     *     sum_k (s/D + (1 - s) y_v) log(s/D + (1 - s) y_v)    */
    real_t loss; 

    /* weights on vertices for losses other than linear (ignored for linear
     * loss); array of length V, or null for no weights */
    const real_t *loss_weights;

    /**  preconditioning and auxiliary variables  **/

    real_t *KL_Ga_Y; // precompute some information for KL

    /**  specialization of base virtual methods  **/

    /* hessian of separable loss */
    void compute_hess_f() override;

    /* compute Lipschitz metric of quadratic functional */
    void compute_lipschitz_metric() override;

    /* compute the gradient of the separable functional in Pfdr::Ga_grad_f */
    void compute_Ga_grad_f() override;

    void compute_prox_Ga_h() override; // backward step over iterate X

    real_t compute_f() override; // separable loss

    void preconditioning(bool init) override; // add some precomputations

    /* relative iterate evolution in l1 norm and components saturation */
    real_t compute_evolution() override;

    /**  type resolution for base template class members  **/
    using Pfdr_d1<real_t, vertex_t>::V;
    using Pfdr_d1<real_t, vertex_t>::E;
    using Pfdr_d1<real_t, vertex_t>::D11;
    using Pfdr<real_t, vertex_t>::Ga_grad_f;
    using Pfdr<real_t, vertex_t>::Ga;
    using Pfdr<real_t, vertex_t>::L;
    using Pfdr<real_t, vertex_t>::l;
    using Pfdr<real_t, vertex_t>::lshape;
    using Pfdr<real_t, vertex_t>::gashape;
    using Pfdr<real_t, vertex_t>::SCALAR;
    using Pfdr<real_t, vertex_t>::MONODIM;
    using Pfdr<real_t, vertex_t>::MULTIDIM;
    using Pfdr<real_t, vertex_t>::lipschcomput;
    using Pfdr<real_t, vertex_t>::Lmut;
    using Pfdr<real_t, vertex_t>::ONCE;
    using Pfdr<real_t, vertex_t>::EACH;
    using Pfdr<real_t, vertex_t>::D;
    using Pcd_prox<real_t>::X;
    using Pcd_prox<real_t>::last_X;
    using Pcd_prox<real_t>::cond_min;
    using Pcd_prox<real_t>::eps;
    using Pcd_prox<real_t>::malloc_check;
};
