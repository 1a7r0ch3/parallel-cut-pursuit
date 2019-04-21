/*=============================================================================
 * Base class for cut-pursuit algorithm
 * 
 * index_t must be able to represent the numbers of vertices and of edges in 
 * the main graph;
 * comp_t must be able to represent the numbers of constant connected 
 * components and of reduced edges in the reduced graph
 *
 * L. Landrieu, L. and G. Obozinski, Cut Pursuit: Fast Algorithms to Learn 
 * Piecewise Constant Functions on General Weighted Graphs, SIAM Journal on 
 * Imaging Sciences, 2017, 10, 1724-1766
 *
 * Hugo Raguet 2018
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *===========================================================================*/
#pragma once
#include <cstdint> // for uintmax_t, requires C++11
#include <cstdlib> // for size_t and malloc
#include <chrono>
#include <limits>
#include <iostream>
#include "cp_graph.hpp" /* Boykov-Kolmogorov graph class modified for CP */
#include "../include/omp_num_threads.hpp"

/* flag an activated edge on residual capacity of its corresponding arcs */
#define ACTIVE_EDGE ((real_t) -1.0) 
#define PAR_SEP_EDGE ((real_t) -2.0) // for parallelization
/* maximum number of components; no component can have this identifier */
#define MAX_NUM_COMP (std::numeric_limits<comp_t>::max())
/* special values for merge step */
#define CHAIN_ROOT MAX_NUM_COMP
#define CHAIN_LEAF MAX_NUM_COMP

/* real_t is the real numeric type, used for objective functional computation
 * and thus for edge weights and flow graph capacities;
 * index_t must be able to represent the number of vertices and of (undirected)
 * edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph;
 * value_t is the type associated to the space to which the values belong, it
 * is usually real_t, and if multidimensional, this must be specified in the
 * parameter D (e.g. for R^3, specify value_t = real_t and D = 3) */
template <typename real_t, typename index_t, typename comp_t,
    typename value_t = real_t>
class Cp
{
public:
    /**  constructor, destructor  **/

    /* only creates flow graph structure */
    Cp(index_t V, index_t E, const index_t* first_edge, 
        const index_t* adj_vertices, size_t D = 1);

    /* the destructor does not free pointers which are supposed to be provided 
     * by the user (forward-star graph structure given at construction, 
     * monitoring arrays, etc.); IT DOES FREE THE REST (components assignment 
     * and reduced problem elements, etc.), but this can be prevented by
     * getting the corresponding pointer member and setting it to null
     * beforehand */
	virtual ~Cp();

    /**  manipulate private members pointers and values  **/

    void reset_edges(); // flag all edges as not active

    /* if 'edge_weights' is null, homogeneously equal to 'homo_edge_weight' */
    void set_edge_weights(const real_t* edge_weights = nullptr,
        real_t homo_edge_weight = 1.0);

    void set_monitoring_arrays(real_t* objective_values = nullptr,
        double* elapsed_time = nullptr, real_t* iterate_evolution = nullptr);

    /* if rV is zero or unity, comp_assign will be automatically initialized;
     * if rV is zero, arbitrary components will be assigned at initialization,
     * in an attempt to optimize parallelization along components;
     * if rV is greater than one, comp_assign must be given and initialized;
     * comp_assign is free()'d by destructor, unless set to null beforehand */
    void set_components(comp_t rV = 0, comp_t* comp_assign = nullptr);

    void set_cp_param(real_t dif_tol, int it_max, int verbose, real_t eps);
    /* overload for default eps parameter */
    void set_cp_param(real_t dif_tol = 0.0, int it_max = 10,
        int verbose = 1000)
    {
        set_cp_param(dif_tol, it_max, verbose,
            std::numeric_limits<real_t>::epsilon());
    }

    void set_parallel_param(int max_num_threads,
        bool balance_par_split = true);
    /* overload for default max_num_threads paramter */
    void set_parallel_param(bool balance_par_split)
    {
        set_parallel_param(omp_get_max_threads(), balance_par_split);
    }

    /* the 'get' methods takes pointers to pointers as arguments; a null means
     * that the user is not interested by the corresponding pointer; NOTA:
     * 1) if not explicitely set by the user, memory pointed by these members
     * is allocated using malloc(), and thus should be deleted with free()
     * 2) they are free()'d by destructor, unless set to null beforehand */
    comp_t get_components(comp_t** comp_assign = nullptr,
        index_t** first_vertex = nullptr, index_t** comp_list = nullptr);

    size_t get_reduced_graph(comp_t** reduced_edges = nullptr,
        real_t** reduced_edge_weights = nullptr);

    /* retrieve the reduced iterate (values of the components);
     * WARNING: reduced values are free()'d by destructor */
    value_t* get_reduced_values();

    /* set the reduced iterate (values of the components);
     * WARNING: if not set to null before deletion of the main cp object,
     * this will be deleted by free() so the given pointer must have been
     * allocated with malloc() and the likes */
    void set_reduced_values(value_t* rX);

    /* solve the main problem */
    int cut_pursuit(bool init = true);

protected:
    /**  main graph  **/

    const index_t V, E; // number of vertices, of edges
    /* forward-star representation:
     * - edges are numeroted so that all vertices originating from a same 
     * vertex are consecutive;
     * - for each vertex, 'first_edge' indicates the first edge starting
     * from the vertex (or, if there are none, starting from the next vertex);
     * array of length V + 1, the first value is always zero and the the last
     * value is always the total number of edges
     * - for each edge, 'adj_vertices' indicates its ending vertex */
    const index_t *first_edge, *adj_vertices; 
    const real_t *edge_weights;
    real_t homo_edge_weight;

    const size_t D; // dimension of the data; total size is V*D
    value_t *rX, *last_rX; // reduced iterate (values of the components)
    comp_t saturated_comp; // number of saturated components
    comp_t saturated_vert; // number of saturated components

    /**  reduced graph  **/

    comp_t rV, last_rV; // number of components (reduced vertices)
    size_t rE; // number of reduced edges
    comp_t* comp_assign; // assignment of each vertex to a component
    /* list the vertices of each components:
     * - vertices are gathered in 'comp_list' so that all vertices belonging
     * to a same components are consecutive
     * - for each component, 'first_vertex' indicates the index of its first
     * vertex in 'comp_list' */
    index_t *comp_list, *first_vertex;
    comp_t* reduced_edges; // array with pair of vertices
    real_t* reduced_edge_weights;

    /** parameters **/

    real_t dif_tol, eps; // eps gives a characteristic precision 
    /* with nonzero verbose information on the process will be printed;
     * for convex methods, this will be passed on to the reduced problem
     * subroutine, controlling the number of subiterations between prints */
    int verbose; 

    /* for stopping criterion or component saturation */
    bool monitor_evolution;

    /**  methods for manipulating nodes and arcs in the flow graph  **/

    bool is_active(index_t e); // check if edge e is active

    bool is_par_sep(index_t e); // check if edge e is a parallel cut separation

    bool is_free(index_t e); // check if edge e is not active or parallel cut 

    void set_active(index_t e); // flag an active edge

    void set_par_sep(index_t e); // flag a parallel cut separation

    void set_inactive(index_t e); // flag an inactive edge

    bool is_sink(index_t v); // check if vertex v is in the sink after min cut

    /* temporary components list and assignment in the flow graph graph */
    comp_t& tmp_comp_assign(index_t v);
    index_t& tmp_comp_list(index_t i);

    /* NOTA: saturation is flagged on the first vertex of the component, so
     * this must be reset if the component list is modified or reordered */
    bool& saturation(comp_t rv); // check component's saturation

    /* manipulate flow graph residual capacities */
    void set_edge_capacities(index_t e, real_t cap_uv, real_t cap_vu);

    real_t& term_capacities(index_t v);

    /**  split components with graph cuts and activate edges, in parallel  **/

    /* split large components for balancing parallel workload:
     * components are split only by adding elements in the first_vertex list;
     * comp_list remains unchanged; new component created this way will be 
     * coherent, as the component list is computed with breadth-first search;
     * rV_new is the number of components resulting from such split;
     * rV_big is the number of large original components split this way;
     * first_vertex_big holds the first vertices of components split this way;
     * return the number of useful parallel threads */
    int balance_parallel_split(comp_t& rV_new, comp_t& rV_big, 
        index_t*& first_vertex_big);

    /* revert the above process;
     * parallel separation edges must be removed or activated */
    virtual index_t remove_parallel_separations(comp_t rV_new) = 0;
    index_t revert_balance_parallel_split(comp_t rV_new, comp_t rV_big,
        index_t* first_vertex_big); // return the number of activations

    /* rough estimate of the number of operations for split step;
     * useful for estimating the number of parallel threads */
    uintmax_t maxflow_complexity(); // just for a graph cut
    virtual uintmax_t split_complexity() = 0;

    /* get a parallel copy of the flow graph */
    Cp_graph<real_t, index_t, comp_t>* get_parallel_flow_graph();

    /* prefered alternative value for each vertex */
    comp_t*& label_assign = comp_assign; // reuse the same storage

    virtual void split_component(Cp_graph<real_t, index_t, comp_t>* G,
        comp_t rv) = 0;

    virtual index_t split();

    /**  compute reduced values  **/

    virtual void solve_reduced_problem() = 0;

    /**  merging components when deemed useful  **/

    /* during the merging step, merged components are stored as chains,
     * represented by arrays of length rV 'merge_chains_root', '_next' and
     * '_leaf'; merge chain involving component rv follows the scheme
     *   root[rv] -> ... -> rv -> next[rv] -> ... -> leaf[rv] ;
     * NOTA: macros CHAIN_ROOT and CHAIN_LEAF are special values, and:
     * - only next[rv] is always up-to-date;
     * - root[rv] is always a preceding component in its chain, or CHAIN_ROOT
     *   if rv is a root;
     * - leaf[rv] is up-to-date if rv is a root;
     * - rv is the leaf of its chain if, and only if next[rv] = CHAIN_LEAF;
     * an additional requirement is that the root of each chain should be the
     * component in the chain with lowest index */
    comp_t get_merge_chain_root(comp_t rv);

    /* merge the merge chains of the two given roots;
     * the root of the resulting chain will be the component in the chains
     * with lowest index, and assigned to the parameter ru; the root of the
     * other chain in the merge is assigned to rv */
    void merge_components(comp_t& ru, comp_t& rv);

    /* compute the merge chains and return the number of effective merges;
     * NOTA: the chosen reduced graph structure is just a list of edges,
     * and does not provide the complete list of edges linking to a given
     * vertex, thus getting back to the root of the chains for each edge is
     * O(rE^2), but is expected to be much less in practice */
    virtual comp_t compute_merge_chains() = 0;

    /* main routine using the above to perform the merge step */
    virtual index_t merge();

    /**  monitoring evolution; set monitor_evolution to true  **/

    /* compute relative iterate evolution;
     * for continuously differentiable problems, saturation is tested here */
    virtual real_t compute_evolution(bool compute_dif) = 0;

    /* compute objective functional, often on the reduced problem objects */
    virtual real_t compute_objective() = 0;

    /* allocate memory and fail with error message if not successful */
    static void* malloc_check(size_t size){
        void *ptr = malloc(size);
        if (!ptr){
            std::cerr << "Cut-pursuit: not enough memory." << std::endl;
            exit(EXIT_FAILURE);
        }
        return ptr;
    }

    /* simply free if size is zero */
    static void* realloc_check(void* ptr, size_t size){
        if (!size){
           free(ptr); 
           return nullptr; 
        }
        ptr = realloc(ptr, size);
        if (!ptr){
            std::cerr << "Cut-pursuit: not enough memory." << std::endl;
            exit(EXIT_FAILURE);
        }
        return ptr;
    }

    /**  control parallelization  **/
    int max_num_threads; // maximum number of parallel threads 
    /* take into account max_num_threads attribute */
    int compute_num_threads(uintmax_t num_ops, uintmax_t max_threads);
    /* overload for max_threads defaulting to num_ops */
    int compute_num_threads(uintmax_t num_ops)
    { return compute_num_threads(num_ops, num_ops); }

private:
    /* parameters */
    int it_max; // maximum number of cut-pursuit iterations
    bool balance_par_split; // switch controling parallel split balancing

    using arc = typename Cp_graph<real_t, index_t, comp_t>::arc;

    Cp_graph<real_t, index_t, comp_t>* G; // flow graph

    /* monitoring */
    real_t* objective_values;
    double* elapsed_time;
    real_t* iterate_evolution;

    double monitor_time(std::chrono::steady_clock::time_point start);

    void print_progress(int it, real_t dif, double t);

    /* set components assignment and values (and allocate them if needed);
     * assumes that no edge of the graph are active when it is called */
    void initialize();

    /* initialize with components specified in 'comp_assign' */
    void assign_connected_components();

    /* initialize with only one component and reduced graph accordingly */
    void single_connected_component();

    /* update connected components and count saturated ones */
    void compute_connected_components();

    /* allocate and compute reduced graph structure */
    void compute_reduced_graph();

    /* during the merging step, merged components are stored as chains */
    comp_t *merge_chains_root, *merge_chains_next, *merge_chains_leaf;
};

#define TPL template <typename real_t, typename index_t, typename comp_t, \
    typename value_t>
#define CP Cp<real_t, index_t, comp_t, value_t>

/***  inline methods in relation with main graph  ***/

TPL inline bool CP::is_active(index_t e)
{ return G->arcs[(size_t) 2*e].r_cap == ACTIVE_EDGE; }

TPL inline bool CP::is_par_sep(index_t e)
{ return G->arcs[(size_t) 2*e].r_cap == PAR_SEP_EDGE; }

TPL inline bool CP::is_free(index_t e)
{ return G->arcs[(size_t) 2*e].r_cap >= 0.0; }

TPL inline bool CP::is_sink(index_t v)
{ return G->nodes[v].is_sink; }

TPL inline comp_t& CP::tmp_comp_assign(index_t v)
{ return G->nodes[v].comp; }

TPL inline index_t& CP::tmp_comp_list(index_t i)
{ return G->nodes[i].vertex; }

TPL inline void CP::set_edge_capacities(index_t e, real_t cap_uv,
    real_t cap_vu)
{
    size_t a = (size_t) 2*e; // cast as size_t to avoid overflow
    G->arcs[a].r_cap = cap_uv;
    G->arcs[a + 1].r_cap = cap_vu;
}

TPL inline void CP::set_active(index_t e)
{ set_edge_capacities(e, ACTIVE_EDGE, ACTIVE_EDGE); }

TPL inline void CP::set_par_sep(index_t e)
{ set_edge_capacities(e, PAR_SEP_EDGE, PAR_SEP_EDGE); }

TPL inline void CP::set_inactive(index_t e)
{ set_edge_capacities(e, 0.0, 0.0); }

TPL inline real_t& CP::term_capacities(index_t v)
{ return G->nodes[v].tr_cap; }

TPL inline Cp_graph<real_t, index_t, comp_t>* CP::get_parallel_flow_graph()
{ return new Cp_graph<real_t, index_t, comp_t>(*G); }

TPL inline bool& CP::saturation(comp_t rv)
{ return G->nodes[comp_list[first_vertex[rv]]].saturation; }

TPL inline comp_t CP::get_merge_chain_root(comp_t rv)
{
    while (merge_chains_root[rv] != CHAIN_ROOT){ rv = merge_chains_root[rv]; }
    return rv;
}

TPL inline int CP::compute_num_threads(uintmax_t num_ops,
    uintmax_t max_threads)
{
    int num_threads = ::compute_num_threads(num_ops, max_threads);
    return num_threads < max_num_threads ? num_threads : max_num_threads;
}

#undef TPL
#undef CP
