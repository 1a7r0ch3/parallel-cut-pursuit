/*=============================================================================
 * Hugo Raguet 2019
 *===========================================================================*/
#include "../include/cut_pursuit_d0.hpp"
#include "../include/omp_num_threads.hpp"

#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)
#define TWO ((real_t) 2.0)
#define EDGE_WEIGHTS_(e) (edge_weights ? edge_weights[(e)] : homo_edge_weight)
/* special flag */
#define MERGE_INIT MAX_NUM_COMP

#define TPL template <typename real_t, typename index_t, typename comp_t, \
    typename value_t>
#define CP_D0 Cp_d0<real_t, index_t, comp_t, value_t>

using namespace std;

TPL CP_D0::Cp_d0(index_t V, index_t E, const index_t* first_edge,
    const index_t* adj_vertices, size_t D) :
    Cp<real_t, index_t, comp_t>(V, E, first_edge, adj_vertices, D),
    no_merge_info(&reserved_merge_info)
{
    K = 2;
    split_iter_num = 2;
}

TPL void CP_D0::set_split_param(int K, int split_iter_num)
{
    if (split_iter_num < 1){
        cerr << "Cut-pursuit d0: there must be at least one iteration in the "
            "split (" << split_iter_num << " specified)." << endl;
        exit(EXIT_FAILURE);
    }

    if (K < 2){
        cerr << "Cut-pursuit d0: there must be at least two alternative values"
            "in the split (" << K << " specified)." << endl;
        exit(EXIT_FAILURE);
    }else if (numeric_limits<comp_t>::max() < K){
        cerr << "Cut-pursuit d0: comp_t must be able to represent the number"
            " of alternative values in the split K (" << K << ")." << endl;
        exit(EXIT_FAILURE);
    }
}

TPL real_t CP_D0::compute_graph_d0()
{
    real_t weighted_contour_length = ZERO;
    #pragma omp parallel for schedule(static) NUM_THREADS(rE) \
        reduction(+:weighted_contour_length)
    for (size_t re = 0; re < rE; re++){
        weighted_contour_length += reduced_edge_weights[re];
    }
    return weighted_contour_length;
}

TPL real_t CP_D0::compute_f()
{
    real_t f = ZERO;
    #pragma omp parallel for schedule(dynamic) NUM_THREADS(D*V, rV) \
        reduction(+:f)
    for (comp_t rv = 0; rv < rV; rv++){
        real_t* rXv = rX + D*rv;
        for (index_t v = first_vertex[rv]; v < first_vertex[rv + 1]; v++){
            f += fv(comp_list[v], rXv);
        }
    }
    return f;
}

TPL real_t CP_D0::compute_objective()
{ return compute_f() + compute_graph_d0(); } // f(x) + ||x||_d0 }

TPL index_t CP_D0::split()
{
    index_t activation = 0;

    /* best alternative label stored temporarily in array 'comp_assign' */
    comp_t* label_assign = comp_assign;

    /**  refine components in parallel  **/
    #pragma omp parallel NUM_THREADS((K - 1)*(2*D*V + 5*E), rV)
    {

    Cp_graph<real_t, index_t, comp_t>* Gpar = get_parallel_flow_graph();
    value_t* altX = (value_t*) malloc_check(sizeof(value_t)*D*K);

    #pragma omp for schedule(dynamic) reduction(+:activation)
    for (comp_t rv = 0; rv < rV; rv++){
        if (is_saturated(rv)){ continue; }
        index_t rv_activation = 0;

        for (int split_it = 0; split_it < split_iter_num; split_it++){

            if (split_it == 0){ init_split_values(rv, altX, label_assign); }
            else{ update_split_values(rv, altX, label_assign); }
    
            bool no_reassignment = true;

            if (K == 2){ /* one graph cut is enough */
                for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1];
                    i++){
                    index_t v = comp_list[i];
                    /* unary cost for chosing the second alternative */
                    set_term_capacities(v, fv(v, altX + D) - fv(v, altX));
                }

                /* set d1 edge capacities within each component */
                for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1];
                    i++){
                    index_t v = comp_list[i];
                    for (index_t e = first_edge[v]; e < first_edge[v + 1];
                        e++){
                        if (!is_active(e)){
                            set_edge_capacities(e, EDGE_WEIGHTS_(e),
                                EDGE_WEIGHTS_(e));
                        }
                    }
                }

                /* find min cut and set assignment accordingly */
                Gpar->maxflow(first_vertex[rv + 1] - first_vertex[rv],
                    comp_list + first_vertex[rv]);

                for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1];
                    i++){
                    index_t v = comp_list[i];
                    if (is_sink(v) != label_assign[v]){
                        label_assign[v] = is_sink(v);
                        no_reassignment = false;
                    }
                }

            }else{ /* iterate over all K alternative values */
                for (comp_t k = 0; k < K; k++){
        
                /* check if alternative k has still vertices assigned to it */
                if (!is_split_value(altX[D*k])){ continue; }

                /* set the source/sink capacities */
                bool all_assigned_k = true;
                for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1];
                    i++){
                    index_t v = comp_list[i];
                    comp_t l = label_assign[v];
                    /* unary cost for changing current value to k-th value */
                    if (l == k){
                        set_term_capacities(v, ZERO);
                    }else{
                        set_term_capacities(v, fv(v, altX + D*k) -
                            fv(v, altX + D*l));
                        all_assigned_k = false;
                    }
                }
                if (all_assigned_k){ continue; }

                /* set d1 edge capacities within each component */
                for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1];
                    i++){
                    index_t u = comp_list[i];
                    comp_t lu = label_assign[u];
                    for (index_t e = first_edge[u]; e < first_edge[u + 1];
                        e++){
                        if (is_active(e)){ continue; }
                        index_t v = adj_vertices[e];
                        comp_t lv = label_assign[v];
                    /* horizontal and source/sink capacities are modified 
                     * according to Kolmogorov & Zabih (2004); in their
                     * notations, functional E(u,v) is decomposed as
                     *
                     * E(0,0) | E(0,1)    A | B
                     * --------------- = -------
                     * E(1,0) | E(1,1)    C | D
                     *                         0 | 0      0 | D-C    0 |B+C-A-D
                     *                 = A + --------- + -------- + -----------
                     *                       C-A | C-A    0 | D-C    0 |   0
                     *
                     *            constant +      unary terms     + binary term
                     */
                        /* A = E(0,0) is the cost of the current assignment */
                        real_t A = lu == lv ? ZERO : EDGE_WEIGHTS_(e);
                        /* B = E(0,1) is the cost of changing lv to k */
                        real_t B = lu == k ? ZERO : EDGE_WEIGHTS_(e);
                        /* C = E(1,0) is the cost of changing lu to k */
                        real_t C = lv == k ? ZERO : EDGE_WEIGHTS_(e);
                        /* D = E(1,1) = 0 is for changing both lu, lv to k */
                        /* set weights in accordance with orientation u -> v */
                        add_term_capacities(u, C - A);
                        add_term_capacities(v, -C);
                        set_edge_capacities(e, B + C - A, ZERO);
                    }
                }

                /* find min cut and update assignment accordingly */
                Gpar->maxflow(first_vertex[rv + 1] - first_vertex[rv],
                    comp_list + first_vertex[rv]);

                for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1];
                    i++){
                    index_t v = comp_list[i];
                    if (is_sink(v) && label_assign[v] != k){
                        label_assign[v] = k;
                        no_reassignment = false;
                    }
                }

                } // end for k
            } // end if K == 2

            if (no_reassignment){ break; }
    
        } // end for split_it

        /* activate edges correspondingly */
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
                if (!is_active(e) &&
                    label_assign[v] != label_assign[adj_vertices[e]]){
                    set_active(e);
                    rv_activation++;
                }
            }
        }

        set_saturation(rv, rv_activation == 0);
        activation += rv_activation;

        /* reconstruct comp_assign */
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            comp_assign[comp_list[i]] = rv;
        }

    } // end for rv

    free(altX);
    delete Gpar;

    } // end parallel region
    
    return activation;
}

TPL CP_D0::Merge_info::Merge_info(size_t D)
{ value = (value_t*) malloc_check(sizeof(value_t)*D); }

TPL CP_D0::Merge_info::~Merge_info()
{ free(value); }

TPL void CP_D0::delete_merge_candidate(size_t re)
{
    if (merge_info_list[re] != no_merge_info){ delete merge_info_list[re]; }
    merge_info_list[re] = nullptr;
}

TPL void CP_D0::select_best_merge_candidate(size_t re, real_t* best_gain,
    index_t* best_edge)
{
    if (merge_info_list[re] != no_merge_info
        && merge_info_list[re]->gain > *best_gain){
            *best_gain = merge_info_list[re]->gain;
            *best_edge = re;
    }
}

TPL void CP_D0::accept_merge_candidate(size_t re, comp_t& ru, comp_t& rv)
{
    merge_components(ru, rv); // ru now the root of the merge chain
    value_t* rXu = rX + D*ru;
    for (size_t d = 0; d < D; d++){ rXu[d] = merge_info_list[re]->value[d]; }
}

TPL comp_t CP_D0::compute_merge_chains()
{
    comp_t merge_count = 0;
   
    merge_info_list = (Merge_info**) malloc_check(sizeof(Merge_info*)*rE);
    for (size_t re = 0; re < rE; re++){ merge_info_list[re] = no_merge_info; }

    real_t* best_par_gains =
        (real_t*) malloc_check(sizeof(real_t)*omp_get_num_procs());
    index_t* best_par_edges = 
        (index_t*) malloc_check(sizeof(index_t)*omp_get_num_procs());

    comp_t last_merge_root = MERGE_INIT;

    while (true){
 
        /**  update merge information in parallel  **/
        int num_par_thrds = last_merge_root == MERGE_INIT ?
            compute_num_threads(update_merge_complexity()) :
            /* expected fraction of merge candidates to update is the total
             * number of edges divided by the expected number of edges linking
             * to the last merged component; in turn, this is estimated as
             * twice the number of edges divided by the number of components */
            compute_num_threads(update_merge_complexity()/rV*2);

        for (int thrd_num = 0; thrd_num < num_par_thrds; thrd_num++){
            best_par_gains[thrd_num] = ZERO;
        }

        /* differences between threads is small: using static schedule */
        #pragma omp parallel for schedule(static) num_threads(num_par_thrds)
        for (size_t re = 0; re < rE; re++){
            if (!merge_info_list[re]){ continue; }
            comp_t ru = reduced_edges[2*re];
            comp_t rv = reduced_edges[2*re + 1];

            if (last_merge_root != MERGE_INIT){
                /* the roots of their respective chains might have changed */
                ru = get_merge_chain_root(ru);
                rv = get_merge_chain_root(rv);
                /* check if none of them is concerned by the last merge */
                if (last_merge_root != ru && last_merge_root != rv){
                    select_best_merge_candidate(re,
                        best_par_gains + omp_get_thread_num(),
                        best_par_edges + omp_get_thread_num());
                    continue;
                }
            }

            if (ru == rv){ /* already merged */
                delete_merge_candidate(re);
            }else{ /* update information */
                update_merge_candidate(re, ru, rv);
                select_best_merge_candidate(re,
                    best_par_gains + omp_get_thread_num(),
                    best_par_edges + omp_get_thread_num());
            }
        } // end for candidates in parallel

        /**  select best candidate  **/
        real_t best_gain = best_par_gains[0];
        size_t best_edge = best_par_edges[0];
        for (int thrd_num = 1; thrd_num < num_par_thrds; thrd_num++){
            if (best_gain < best_par_gains[thrd_num]){
                best_gain = best_par_gains[thrd_num];
                best_edge = best_par_edges[thrd_num];
            }
        }

        /**  merge best candidate if best gain is positive  **/
        if (best_gain > ZERO){
            comp_t ru = get_merge_chain_root(reduced_edges[2*best_edge]);
            comp_t rv = get_merge_chain_root(reduced_edges[2*best_edge + 1]);
            accept_merge_candidate(best_edge, ru, rv); // ru now the root
            delete_merge_candidate(best_edge);
            merge_count++;
            last_merge_root = ru;
        }else{
            break;
        }
   
    } // end merge loop

    free(best_par_gains);
    free(best_par_edges);
    free(merge_info_list); // all merge info must have been deleted

    return merge_count;
}

/**  instantiate for compilation  **/
template class Cp_d0<float, uint32_t, uint16_t>;
template class Cp_d0<double, uint32_t, uint16_t>;
template class Cp_d0<float, uint32_t, uint32_t>;
template class Cp_d0<double, uint32_t, uint32_t>;
