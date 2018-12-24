/*=============================================================================
 * Hugo Raguet 2018
 *===========================================================================*/
#include <cmath>
#include "../include/cut_pursuit_d1.hpp"
#include "../include/omp_num_threads.hpp"

#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)
#define COOR_WEIGHTS_(d) (coor_weights ? coor_weights[(d)] : ONE)

using namespace std;

template <typename real_t, typename index_t, typename comp_t>
Cp_d1<real_t, index_t, comp_t>::Cp_d1(index_t V, size_t D, index_t E,
    const index_t* first_edge, const index_t* adj_vertices, D1p d1p) :
    Cp<real_t, index_t, comp_t>(V, D, E, first_edge, adj_vertices), d1p(d1p)
{ coor_weights = nullptr; }

template <typename real_t, typename index_t, typename comp_t>
void Cp_d1<real_t, index_t, comp_t>::set_edge_weights(const real_t* edge_weights,
    real_t homo_edge_weight, const real_t* coor_weights)
{
    Cp<real_t, index_t, comp_t>::set_edge_weights(edge_weights,
        homo_edge_weight);
    this->coor_weights = coor_weights;
}

template <typename real_t, typename index_t, typename comp_t>
bool Cp_d1<real_t, index_t, comp_t>::is_almost_equal(comp_t ru, comp_t rv)
{
    real_t dif = ZERO, ampu = ZERO, ampv = ZERO;
    real_t *rXu = rX + ru*D;
    real_t *rXv = rX + rv*D;
    for (size_t d = 0; d < D; d++){
        if (d1p == D11){
            dif += abs(rXu[d] - rXv[d])*COOR_WEIGHTS_(d);
            ampu += abs(rXu[d])*COOR_WEIGHTS_(d);
            ampv += abs(rXv[d])*COOR_WEIGHTS_(d);
        }else if (d1p == D12){
            dif += (rXu[d] - rXv[d])*(rXu[d] - rXv[d])*COOR_WEIGHTS_(d);
            ampu += rXu[d]*rXu[d]*COOR_WEIGHTS_(d);
            ampv += rXv[d]*rXv[d]*COOR_WEIGHTS_(d);
        }
    }
    real_t amp = ampu > ampv ? ampu : ampv;
    if (d1p == D12){ dif = sqrt(dif); amp = sqrt(amp); }
    if (eps > amp){ amp = eps; }
    return dif <= dif_tol*amp;
}

template <typename real_t, typename index_t, typename comp_t> 
index_t Cp_d1<real_t, index_t, comp_t>::merge()
{
    index_t deactivation = 0;

    /* compute connected components of the reduced graph, stored as chains
     * root[v] -> ... -> v -> next[v] -> ... -> leaf[v] 
     * the chosen reduced graph structure does not provide the complete list of 
     * edges starting or ending at a given vertex, so this is this is O(rE^2)
     * in the worst case, but is expected to be much less in practice */
    comp_t* root = (comp_t*) malloc_check(rV*sizeof(comp_t)); 
    comp_t* next = (comp_t*) malloc_check(rV*sizeof(comp_t));
    comp_t* leaf = (comp_t*) malloc_check(rV*sizeof(comp_t));
    for (comp_t rv = 0; rv < rV; rv++){ root[rv] = next[rv] = leaf[rv] = rv; }

    comp_t merges = 0;
    for (size_t re = 0; re < rE; re++){
        comp_t ru = reduced_edges[2*re];
        comp_t rv = reduced_edges[2*re + 1];
        if (ru != rv && is_almost_equal(ru, rv)){ /* merge ru and rv */
            merges++;
            /* get back to the roots of their respective chains */
            comp_t ur = ru; while (ur != root[ur]){ ur = root[ur]; }
            comp_t vr = rv; while (vr != root[vr]){ vr = root[vr]; }
            /* ensure the smallest component is the root of the merge chain */
            if (ur > vr){ comp_t tmp = ur; ur = vr; vr = tmp; }
            if (ur != vr){
                next[leaf[ur]] = vr; // link both chains
                leaf[ur] = leaf[vr]; // update leaf of the chain
            }
            /* update root information */
            root[ru] = root[leaf[ur]] = root[vr] = root[rv] = root[leaf[vr]]
                = ur;
            set_saturation(ur, false); // merged components are not saturated
        }
    }

    if (!merges){
        free(root); free(next); free(leaf);

        return deactivation;
    }

    /* construct a new version of 'comp_list' in temporary storage; update
     * 'rX', 'first_vertex', 'reduced_edges' and '_weights' in-place */
    comp_t* new_comp_id = leaf; // leaf will not be used anymore
    comp_t rn = 0; // component number
    index_t i = 0; // index in the new comp_list
    for (comp_t ru = 0; ru < rV; ru++){
        if (root[ru] != ru){ continue; }
        comp_t rv = ru; // it's a root
        /* rX can be modified in-place since rn <= ru */
        real_t *rXn = rX + D*rn;
        real_t *rXu = rX + D*ru;
        for (size_t d = 0; d < D; d++){ rXn[d] = rXu[d]; }
        index_t first = i; // holds index of first vertex of the component
        while (true){
            new_comp_id[rv] = rn;
            for (index_t v = first_vertex[rv]; v < first_vertex[rv+1]; v++){
                set_tmp_comp_list(i++, comp_list[v]);
            }
            if (next[rv] == rv){ break; } // end of chain
            else{ rv = next[rv]; }
        }
        /* the root of each chain is the smallest component contained in the
         * chain, so now that 'rn' new components have been constructed, the
         * first 'rn' previous components have been copied, hence
         * 'first_vertex' will not be accessed before position 'rn' anymore;
         * it can thus be modified in-place; */
        first_vertex[rn++] = first;
    }
    first_vertex[rV = rn] = V;
    first_vertex = (index_t*) realloc_check(first_vertex,
        sizeof(index_t)*(rV + 1));
    rX = (real_t*) realloc_check(rX, sizeof(real_t)*D*rV);
    /* update components assignments */
    for (index_t v = 0; v < V; v++){
        comp_list[v] = get_tmp_comp_list(v);
        comp_assign[v] = new_comp_id[comp_assign[v]];
    }

    /* update corresponding reduced edges */
    size_t new_re = 0;
    for (size_t re = 0; re < rE; re++){
        comp_t new_ru = new_comp_id[reduced_edges[2*re]];
        comp_t new_rv = new_comp_id[reduced_edges[2*re + 1]];
        if (new_ru != new_rv){
            reduced_edges[2*new_re] = new_ru;
            reduced_edges[2*new_re + 1] = new_rv;
            reduced_edge_weights[new_re] = reduced_edge_weights[re];
            new_re++;
        }
    }
    rE = new_re; // actually only upper bound, some edges might appear twice
    reduced_edges = (comp_t*) realloc_check(reduced_edges,
            sizeof(comp_t)*2*rE);
    reduced_edge_weights = (real_t*) realloc_check(reduced_edge_weights,
            sizeof(real_t)*rE);

    /* deactivate corresponding edges */
    #pragma omp parallel for schedule(dynamic) NUM_THREADS(E, V)
    for (index_t v = 0; v < V; v++){ /* will run along all edges */
        comp_t rv = comp_assign[v];
        for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
            if (is_active(e) && rv == comp_assign[adj_vertices[e]]){
                set_inactive(e);
                deactivation++;
            }
        }
    }

    free(root); free(next); free(leaf);

    return deactivation;
}

template <typename real_t, typename index_t, typename comp_t>
real_t Cp_d1<real_t, index_t, comp_t>::compute_graph_d1()
{
    real_t tv = ZERO;
    #pragma omp parallel for schedule(static) NUM_THREADS(2*E*D, E) \
        reduction(+:tv)
    for (size_t re = 0; re < rE; re++){
        real_t *rXu = rX + reduced_edges[2*re]*D;
        real_t *rXv = rX + reduced_edges[2*re + 1]*D;
        real_t dif = ZERO;
        for (size_t d = 0; d < D; d++){
            if (d1p == D11){
                dif += abs(rXu[d] - rXv[d])*COOR_WEIGHTS_(d);
            }else if (d1p == D12){
                dif += (rXu[d] - rXv[d])*(rXu[d] - rXv[d])*COOR_WEIGHTS_(d);
            }
        }
        if (d1p == D12){ dif = sqrt(dif); }
        tv += reduced_edge_weights[re]*dif;
    }
    return tv;
}

/**  instantiate for compilation  **/
template class Cp_d1<float, uint32_t, uint16_t>;

template class Cp_d1<double, uint32_t, uint16_t>;

template class Cp_d1<float, uint32_t, uint32_t>;

template class Cp_d1<double, uint32_t, uint32_t>;
