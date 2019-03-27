/*=============================================================================
 * Derived class for cut-pursuit algorithm with d1 (total variation) 
 * penalization
 *
 * Reference: H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
 * Nonsmooth Functionals with Graph Total Variation.
 *
 * Hugo Raguet 2018
 *============================================================================*/
#pragma once
#include "cut_pursuit.hpp"

/* real_t is the real numeric type, used for the base field and for the
 * objective functional computation;
 * index_t must be able to represent the numbers of vertices and of
 * (undirected) edges in the main graph; comp_t must be able to represent one
 * plus the number of constant connected components in the reduced graph */
template <typename real_t, typename index_t, typename comp_t>
class Cp_d1 : public Cp<real_t, index_t, comp_t>
{
public:
    /* for multidimensional data, type of graph total variation, which is
     * nothing but the sum of lp norms of finite differences over the edges:
     * d1,1 is the sum of l1 norms;
     * d1,2 is the sum of l2 norms */
    enum D1p {D11, D12};

    Cp_d1(index_t V, index_t E, const index_t* first_edge, 
        const index_t* adj_vertices, size_t D, D1p d1p = D12);

    /* delegation for monodimensional setting */
    Cp_d1(index_t V, index_t E, const index_t* first_edge, 
        const index_t* adj_vertices) :
        Cp_d1(V, E, first_edge, adj_vertices, 1, D11){};

    /* the destructor does not free pointers which are supposed to be provided 
     * by the user (forward-star graph structure given at construction, 
     * edge weights, etc.); IT DOES FREE THE REST (components assignment 
     * and reduced problem elements, etc.), but this can be prevented by
     * getting the corresponding pointer member and setting it to null
     * beforehand */

    /* overload allowing for different weights along coordinates;
     * if 'edge_weights' is null, homogeneously equal to 'homo_edge_weight' */
    void set_edge_weights(const real_t* edge_weights = nullptr,
        real_t homo_edge_weight = 1.0, const real_t* coor_weights = nullptr);

protected:
    /* for multidimensional data, weights the coordinates in the lp norms;
     * all weights must be strictly positive, and it is advised to normalize
     * the weights so that the first value is unity */
    const real_t *coor_weights;

    /* compute graph total variation; use reduced edges and reduced weights */
    real_t compute_graph_d1();

    /**  type resolution for base template class members  **/
    using Cp<real_t, index_t, comp_t>::rX;
    using Cp<real_t, index_t, comp_t>::last_rX;
    using Cp<real_t, index_t, comp_t>::eps;
    using Cp<real_t, index_t, comp_t>::dif_tol;
    using Cp<real_t, index_t, comp_t>::V;
    using Cp<real_t, index_t, comp_t>::E;
    using Cp<real_t, index_t, comp_t>::D;
    using Cp<real_t, index_t, comp_t>::first_edge;
    using Cp<real_t, index_t, comp_t>::adj_vertices; 
    using Cp<real_t, index_t, comp_t>::rV;
    using Cp<real_t, index_t, comp_t>::rE;
    using Cp<real_t, index_t, comp_t>::first_vertex;
    using Cp<real_t, index_t, comp_t>::reduced_edge_weights;
    using Cp<real_t, index_t, comp_t>::reduced_edges;
    using Cp<real_t, index_t, comp_t>::malloc_check;
    using Cp<real_t, index_t, comp_t>::realloc_check;

private:
    const D1p d1p; // see public enum declaration

    /* test if two components are sufficiently close to merge */
    bool is_almost_equal(comp_t ru, comp_t rv);

    /* compute the merge chains and return the number of effective merges */
    comp_t compute_merge_chains() override;

    /**  type resolution for base template class members  **/
    using Cp<real_t, index_t, comp_t>::merge_components;
    using Cp<real_t, index_t, comp_t>::get_merge_chain_root;
};
