/*============================================================================
 * Derived class for cut-pursuit algorithm with d1 (total variation) 
 * penalization, with a separable loss term and simplex constraints:
 *
 * minimize functional over a graph G = (V, E)
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
 * using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
 * splitting algorithm.
 *
 * Parallel implementation with OpenMP API.
 *
 * H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing Nonsmooth
 * Functionals with Graph Total Variation, International Conference on Machine
 * Learning, PMLR, 2018, 80, 4244-4253
 *
 * H. Raguet, A Note on the Forward-Douglas--Rachford Splitting for Monotone 
 * Inclusion and Convex Optimization, Optimization Letters, 2018, 1-24
 *
 * Hugo Raguet 2018
 *===========================================================================*/
#pragma once
#include "cut_pursuit_d1.hpp"
/* these macros must correspond with the ones in pfdr_d1_lsx.hpp */
#define LINEAR ((real_t) 0.0)
#define QUADRATIC ((real_t) 1.0)

/* index_t must be able to represent the numbers of vertices and of edges
 * in the main graph; comp_t must be able to represent the numbers of constant 
 * connected components in the reduced graph, and the dimension D */

/* index_t must be able to represent the numbers of vertices and of
 * (undirected) edges in the main graph; comp_t must be able to represent the
 * numbers of constant connected components in the reduced graph, as well as
 * and the dimension D */
template <typename real_t, typename index_t, typename comp_t>
class Cp_d1_lsx : public Cp_d1<real_t, index_t, comp_t>
{
private:
    /**  type resolution for base template class members  **/
    using Cp<real_t, index_t, comp_t>::dif_tol;

public:
    /**  constructor, destructor  **/

    /* only creates BK graph structure and assign Y, D */
    Cp_d1_lsx(index_t V, size_t D, index_t E, const index_t* first_edge,
        const index_t* adj_vertices, const real_t* Y);

    /* the destructor does not free pointers which are supposed to be provided 
     * by the user (forward-star graph structure given at construction, 
     * monitoring arrays, observation arrays); it does free the rest 
     * (components assignment and reduced problem elements, etc.), but this can 
     * be prevented by copying the corresponding pointer member and set it to 
     * null before deleting */

    /**  methods for manipulating parameters  **/

    /* Y is changed only if the corresponding argument is not null */
    void set_loss(real_t loss, const real_t* Y = nullptr,
        const real_t* loss_weights = nullptr);

    /* overload for changing only loss_weights, or the observations Y when the
     * loss is linear (in which case loss_weights is ignored) */
    void set_loss(const real_t* loss_weights_or_Y)
    {
        if (loss == LINEAR){ set_loss(loss, loss_weights_or_Y); }
        else{ set_loss(loss, nullptr, loss_weights); }
    }

    void set_pfdr_param(real_t rho, real_t cond_min, real_t dif_rcd,
        int it_max, real_t dif_tol);

    /* overload for default dif_tol parameter */
    void set_pfdr_param(real_t rho = 1.0, real_t cond_min = 1e-2,
        real_t dif_rcd = 0.0, int it_max = 1e4)
    { set_pfdr_param(rho, cond_min, dif_rcd, it_max, 1e-3*dif_tol); }

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

    /**  reduced problem  **/
    real_t pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol;
    int pfdr_it, pfdr_it_max;

    /**  methods  **/

    /* solve unidimensional loss problem on simplex; in this specialization,
     * will always be called with component 0 containing full graph */
    void solve_univertex_problem(real_t* uX, comp_t rv = 0);

    void solve_reduced_problem();

    index_t split();

    /* relative iterate evolution in l1 norm and components saturation */
    real_t compute_evolution(const bool compute_dif, comp_t & saturation);

    real_t compute_objective();

    /**  type resolution for base template class members  **/
    using Cp_d1<real_t, index_t, comp_t>::compute_graph_d1;
    using Cp_d1<real_t, index_t, comp_t>::coor_weights;
    using Cp_d1<real_t, index_t, comp_t>::D11;
    using Cp<real_t, index_t, comp_t>::monitor_evolution;
    using Cp<real_t, index_t, comp_t>::set_saturation;
    using Cp<real_t, index_t, comp_t>::is_saturated;
    using Cp<real_t, index_t, comp_t>::is_sink;
    using Cp<real_t, index_t, comp_t>::is_active;
    using Cp<real_t, index_t, comp_t>::set_active;
    using Cp<real_t, index_t, comp_t>::set_edge_capacities;
    using Cp<real_t, index_t, comp_t>::set_term_capacities;
    using Cp<real_t, index_t, comp_t>::add_term_capacities;
    using Cp<real_t, index_t, comp_t>::get_tmp_comp_assign;
    using Cp<real_t, index_t, comp_t>::get_parallel_flow_graph;
    using Cp<real_t, index_t, comp_t>::eps;
    using Cp<real_t, index_t, comp_t>::V;
    using Cp<real_t, index_t, comp_t>::D;
    using Cp<real_t, index_t, comp_t>::E;
    using Cp<real_t, index_t, comp_t>::first_edge;
    using Cp<real_t, index_t, comp_t>::adj_vertices; 
    using Cp<real_t, index_t, comp_t>::edge_weights;
    using Cp<real_t, index_t, comp_t>::homo_edge_weight;
    using Cp<real_t, index_t, comp_t>::rV;
    using Cp<real_t, index_t, comp_t>::rE;
    using Cp<real_t, index_t, comp_t>::comp_assign;
    using Cp<real_t, index_t, comp_t>::rX;
    using Cp<real_t, index_t, comp_t>::last_rX;
    using Cp<real_t, index_t, comp_t>::comp_list;
    using Cp<real_t, index_t, comp_t>::first_vertex;
    using Cp<real_t, index_t, comp_t>::reduced_edges;
    using Cp<real_t, index_t, comp_t>::reduced_edge_weights;
    using Cp<real_t, index_t, comp_t>::verbose;
    using Cp<real_t, index_t, comp_t>::malloc_check;
};
