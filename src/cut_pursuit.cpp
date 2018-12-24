/*=============================================================================
 * Hugo Raguet 2018
 *===========================================================================*/
#include <cmath>
#include "../include/cut_pursuit.hpp"
#include "../include/omp_num_threads.hpp"

#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)
#define EDGE_WEIGHTS_(e) (edge_weights ? edge_weights[(e)] : homo_edge_weight)
#define NOT_ASSIGNED ((comp_t) -1)
#define ASSIGNED ((comp_t) 1)
#define ASSIGNED_ROOT ((comp_t) 2)
#define MAX_COMP (std::numeric_limits<comp_t>::max())
#define NO_EDGE ((index_t) -1)

using namespace std;

template <typename real_t, typename index_t, typename comp_t>
Cp<real_t, index_t, comp_t>::Cp(index_t V, size_t D, index_t E,
    const index_t* first_edge, const index_t* adj_vertices)
    : V(V), D(D), E(E), first_edge(first_edge), adj_vertices(adj_vertices)
{
    /* construct graph */
    G = new Cp_graph<real_t, index_t, comp_t>(V, E);
    G->add_node(V);
    /* d1 edges */
    for (index_t v = 0; v < V; v++){ /* will run along all edges */
        for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
            G->add_edge(v, adj_vertices[e], ZERO, ZERO);
        }
    }
    /* source/sink edges does not need to be initialized */

    rV = 1; rE = 0;
    edge_weights = nullptr;
    homo_edge_weight = ONE;
    comp_assign = nullptr;
    comp_list = first_vertex = nullptr;
    rX = reduced_edge_weights = nullptr;
    reduced_edges = nullptr;
    elapsed_time = nullptr;
    objective_values = iterate_evolution = nullptr;
    
    it_max = 10; verbose = 1000;
    dif_tol = eps = numeric_limits<real_t>::epsilon();
    monitor_evolution = false;
}

template <typename real_t, typename index_t, typename comp_t>
Cp<real_t, index_t, comp_t>::~Cp()
{
    delete G;
    free(comp_assign); free(comp_list); free(first_vertex);
    free(rX); free(reduced_edges); free(reduced_edge_weights);
}

template <typename real_t, typename index_t, typename comp_t>
void Cp<real_t, index_t, comp_t>::reset_active_edges()
{ for (index_t e = 0; e < E; e++){ set_inactive(e); } }

template <typename real_t, typename index_t, typename comp_t>
void Cp<real_t, index_t, comp_t>::set_edge_weights(const real_t* edge_weights,
    real_t homo_edge_weight)
{
    this->edge_weights = edge_weights;
    this->homo_edge_weight = homo_edge_weight;
}

template <typename real_t, typename index_t, typename comp_t>
void Cp<real_t, index_t, comp_t>::set_monitoring_arrays(real_t* 
    objective_values, double* elapsed_time, real_t* iterate_evolution)
{
    this->objective_values = objective_values;
    this->elapsed_time = elapsed_time;
    this->iterate_evolution = iterate_evolution;
    if (iterate_evolution){ monitor_evolution = true; }
}

template <typename real_t, typename index_t, typename comp_t>
void Cp<real_t, index_t, comp_t>::set_components(comp_t rV, comp_t* comp_assign)
{
    if (rV > 1 && !comp_assign){
        cerr << "Cut-pursuit: if an initial number of components greater than "
            "unity is given, components assignment should be provided." << endl;
        exit(EXIT_FAILURE);
    }
    this->rV = rV;
    this->comp_assign = comp_assign;
}

template <typename real_t, typename index_t, typename comp_t>
void Cp<real_t, index_t, comp_t>::set_cp_param(real_t dif_tol, int it_max, 
    int verbose, real_t eps)
{
    this->dif_tol = dif_tol;
    if (dif_tol > ZERO){ monitor_evolution = true; }
    this->it_max = it_max;
    this->verbose = verbose;
    this->eps = ZERO < dif_tol && dif_tol < eps ? dif_tol : eps;
}

template <typename real_t, typename index_t, typename comp_t>
comp_t Cp<real_t, index_t, comp_t>::get_components(comp_t** comp_assign,
    index_t** first_vertex, index_t** comp_list)
{
    if (comp_assign){ *comp_assign = this->comp_assign; }
    if (first_vertex){ *first_vertex = this->first_vertex; }
    if (comp_list){ *comp_list = this->comp_list; }
    return this->rV;
}

template <typename real_t, typename index_t, typename comp_t>
real_t* Cp<real_t, index_t, comp_t>::get_reduced_values()
{ return this->rX; }

template <typename real_t, typename index_t, typename comp_t>
size_t Cp<real_t, index_t, comp_t>::get_reduced_graph(comp_t** reduced_edges,
    real_t** reduced_edge_weights)
{
    if (reduced_edges){ *reduced_edges = this->reduced_edges; }
    if (reduced_edge_weights){
        *reduced_edge_weights = this->reduced_edge_weights;
    }
    return this->rE;
}

template <typename real_t, typename index_t, typename comp_t>
int Cp<real_t, index_t, comp_t>::cut_pursuit(bool init)
{
    int it = 0;
    double timer = 0.0;
    real_t dif = dif_tol > ONE ? dif_tol : ONE;
    comp_t saturation = 0;

    chrono::steady_clock::time_point start;
    if (elapsed_time){ start = chrono::steady_clock::now(); }
    if (init){
        if (verbose){ cout << "\tCut-pursuit initialization:" << endl; }
        initialize();
        if (objective_values){ objective_values[0] = compute_objective(); }
    }
    
    while (true){
        if (elapsed_time){ elapsed_time[it] = timer = monitor_time(start); }
        if (verbose){ print_progress(it, dif, saturation, timer); }
        if (it == it_max || dif < dif_tol){ break; }

        if (verbose){ cout << "\tSplit... " << flush; }
        index_t activation = split();
        if (verbose){ cout << activation << " new activated edge(s)." << endl; }

        if (!activation){ /* do not recompute reduced problem */
            if (dif_tol > ZERO || iterate_evolution){
                dif = ZERO;
                if (iterate_evolution){ iterate_evolution[it] = dif; }
            }

            it++;

            if (objective_values){
                objective_values[it] = objective_values[it - 1];
            }
            continue;
        }else{ /* reduced graph and components will be updated */
            if (monitor_evolution){
                /* store last iterate values */
                last_rX = (real_t*) malloc_check(sizeof(real_t)*D*rV);
                for (size_t rvd = 0; rvd < D*rV; rvd++){
                    last_rX[rvd] = rX[rvd];
                }
                /* store previous assignment */
                for (index_t v = 0; v < V; v++){
                    set_tmp_comp_assign(v, comp_assign[v]);
                }
            }
            free(rX);
            free(reduced_edges);
            free(reduced_edge_weights);
        }

        if (verbose){ cout << "\tCompute connected components... " << flush; }
        saturation = compute_connected_components();
        if (verbose){
            cout << rV << " connected component(s); "
                << saturation << " saturated." << endl;
        }

        if (verbose){ cout << "\tCompute reduced graph... " << flush; }
        compute_reduced_graph();
        if (verbose){ cout << rE << " reduced edge(s)." << endl; }

        if (verbose){ cout << "\tSolve reduced problem: " << endl; }
        solve_reduced_problem();

        if (verbose){ cout << "\tMerge... " << flush; }
        index_t deactivation = merge();
        if (verbose){ cout << deactivation << " deactivated edge(s)." << endl; }

        if (monitor_evolution){
            dif = compute_evolution(dif_tol > ZERO || iterate_evolution,
                saturation);
            if (iterate_evolution){ iterate_evolution[it] = dif; }
            free(last_rX);
        }

        it++;

        if (objective_values){ objective_values[it] = compute_objective(); }

    } /* endwhile true */

    return it;
}

template <typename real_t, typename index_t, typename comp_t>
real_t Cp<real_t, index_t, comp_t>::monitor_time(
    chrono::steady_clock::time_point start)
{ 
    using namespace chrono;
    steady_clock::time_point current = steady_clock::now();
    return ((current - start).count()) * steady_clock::period::num
               / static_cast<double>(steady_clock::period::den);
}

template <typename real_t, typename index_t, typename comp_t>
void Cp<real_t, index_t, comp_t>::print_progress(int it, real_t dif,
    comp_t saturation, double timer)
{
    cout << "\n\tCut-pursuit iteration " << it << " (max. " << it_max << ")\n";
    if (dif_tol > ZERO || iterate_evolution){
        cout.precision(2);
        cout << scientific << "\trelative iterate evolution " << dif
            << " (tol. " << dif_tol << ")\n";
    }
    cout << "\t" << rV << " connected component(s), " << saturation <<
        " saturated, and at most " << rE << " reduced edge(s)\n";
    if (timer > 0.0){
        cout.precision(1);
        cout << fixed << "\telapsed time " << fixed << timer << " s\n";
    }
    cout << endl;
}

template <typename real_t, typename index_t, typename comp_t>
void Cp<real_t, index_t, comp_t>::single_connected_component()
{
    for (index_t v = 0; v < V; v++){ comp_assign[v] = 0; }
    first_vertex = (index_t*) malloc_check(sizeof(index_t)*2);
    first_vertex[0] = 0; first_vertex[1] = V;
    for (index_t v = 0; v < V; v++){ comp_list[v] = v; }
    set_saturation(0, false);
    /* reduced graph contains only the edge from the component to itself */
    if (rE > 0){ free(reduced_edges); free(reduced_edge_weights); }
    rE = 1;
    reduced_edges = (comp_t*) malloc_check(sizeof(comp_t)*2);
    reduced_edges[0] = reduced_edges[1] = 0;
    reduced_edge_weights = (real_t*) malloc_check(sizeof(real_t)*1);
    reduced_edge_weights[0] = eps;
}

template <typename real_t, typename index_t, typename comp_t>
void Cp<real_t, index_t, comp_t>::arbitrary_connected_components()
{
    index_t max_comp_size = V/rV + 1;
    first_vertex = (index_t*) malloc_check(sizeof(index_t)*(rV + 1));

    /* cleanup assigned components */
    for (index_t v = 0; v < V; v++){ comp_assign[v] = NOT_ASSIGNED; }

    index_t rv = 0, comp_size = 0; // rv of type index_t to watch for overflow
    first_vertex[0] = 0;
    index_t i = 0, j = 0; // indices in connected components list
    for (index_t u = 0; u < V; u++){
        if (comp_assign[u] != NOT_ASSIGNED){ continue; }
        /* new component starting at u */
        comp_assign[u] = rv;
        comp_size++;
        comp_list[j++] = u;
        while (i < j){
            index_t v = comp_list[i++];
            /* add neighbors to the connected component list */
            for (arc* a = G->nodes[v].first; a; a = a->next){
                index_t w = a->head - G->nodes; // adjacent vertex
                if (comp_assign[w] != NOT_ASSIGNED){
                    if (comp_assign[w] != comp_assign[v]){ // might not be rv
                        a->r_cap = a->sister->r_cap = ACTIVE_EDGE;
                    }
                    continue;
                }
                /* put in current connected component */
                comp_assign[w] = rv;
                comp_size++;
                comp_list[j++] = w;
                if (comp_size == max_comp_size){ // change component
                    if (rv == rV){
                        rV *= 2;
                        first_vertex = (index_t*) realloc_check(first_vertex,
                            sizeof(index_t)*(rV + 1));
                    }
                    first_vertex[rv + 1] = first_vertex[rv] + comp_size;
                    set_saturation(rv, false);
                    rv++; comp_size = 0;
                }
            }
        } // the current connected component cannot grow anymore
        if (comp_size > 0){ // change current component
            if (rv == rV){
                rV *= 2;
                first_vertex = (index_t*) realloc_check(first_vertex,
                    sizeof(index_t)*(rV + 1));
            }
            first_vertex[rv + 1] = first_vertex[rv] + comp_size;
            set_saturation(rv, false);
            rv++; comp_size = 0;
        }
    }
    /* update components lists and assignments */
    if (rv > MAX_COMP){
        cerr << "Cut-pursuit: number of components (" << rv << ") greater "
            << "than can be represented by comp_t (" << MAX_COMP << ")" << endl;
        exit(EXIT_FAILURE);
    }
    if (rV > rv){
        rV = rv;
        first_vertex = (index_t*) realloc_check(first_vertex,
            sizeof(index_t)*(rV + 1));
    }
}

template <typename real_t, typename index_t, typename comp_t>
void Cp<real_t, index_t, comp_t>::assign_connected_components()
{
    /* activate arcs between components */
    #pragma omp parallel for schedule(dynamic) NUM_THREADS(E, V)
    for (index_t v = 0; v < V; v++){ /* will run along all edges */
        comp_t rv = comp_assign[v];
        for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
            if (rv != comp_assign[adj_vertices[e]]){ set_active(e); }
        }
    }
    /* translate 'comp_assign' into dual representation 'comp_list' */
    first_vertex = (index_t*) malloc_check(sizeof(index_t)*(rV + 1));
    for (comp_t rv = 0; rv < rV + 1; rv++){ first_vertex[rv] = 0; }
    for (index_t v = 0; v < V; v++){ first_vertex[comp_assign[v] + 1]++; }
    for (comp_t rv = 1; rv < rV - 1; rv++){
        first_vertex[rv + 1] += first_vertex[rv];
    }
    for (index_t v = 0; v < V; v++){
        comp_list[first_vertex[comp_assign[v]]++] = v;
    }
    for (comp_t rv = rV; rv > 0; rv--){
        first_vertex[rv] = first_vertex[rv - 1];
    }
    first_vertex[0] = 0;
    for (comp_t rv = 0; rv < rV; rv++){ set_saturation(rv, false); }
}

template <typename real_t, typename index_t, typename comp_t>
comp_t Cp<real_t, index_t, comp_t>::compute_connected_components()
{
    comp_t saturation = 0;

    /* cleanup assigned components */
    for (index_t v = 0; v < V; v++){ comp_assign[v] = NOT_ASSIGNED; }

    index_t rVtmp = 0; // identify and count components
    /* new connected components hierarchically derives from the previous ones,
     * we can thus compute them in parallel along previous components */
    #pragma omp parallel for schedule(dynamic) NUM_THREADS(2*E, rV) \
        reduction(+:rVtmp, saturation)
    for (comp_t rv = 0; rv < rV; rv++){
        if (is_saturated(rv)){ // component stays the same
            saturation++;
            index_t i = first_vertex[rv];
            index_t v = comp_list[i];
            comp_assign[v] = ASSIGNED_ROOT; // flag the component's root
            set_tmp_comp_list(i, v);
            for (i++; i < first_vertex[rv + 1]; i++){
                v = comp_list[i];
                comp_assign[v] = ASSIGNED;
                set_tmp_comp_list(i, v);
            }
            rVtmp++;
            continue;
        }
        index_t i, j, k;
        for (i = j = k = first_vertex[rv]; k < first_vertex[rv + 1]; k++){
            index_t u = comp_list[k];
            if (comp_assign[u] != NOT_ASSIGNED){ continue; }
            comp_assign[u] = ASSIGNED_ROOT; // flag a component's root
            G->nodes[u].saturation = false; // a new component is not saturated
            /* put in connected components list */
            set_tmp_comp_list(j++, u);
            while (i < j){
                index_t v = get_tmp_comp_list(i++);
                /* add neighbors to the connected component list */
                for (arc* a = G->nodes[v].first; a; a = a->next){
                    if (a->r_cap != ACTIVE_EDGE){
                        index_t w = a->head - G->nodes; // adjacent vertex
                        if (comp_assign[w] != NOT_ASSIGNED){ continue; }
                        comp_assign[w] = ASSIGNED;
                        set_tmp_comp_list(j++, w);
                    }
                }
            } /* the current connected component is complete */
            rVtmp++;
        }
    }

    /* update components lists and assignments */
    if (rVtmp > MAX_COMP){
        cerr << "Cut-pursuit: number of components (" << rVtmp << ") greater "
            << "than can be represented by comp_t (" << MAX_COMP << ")" << endl;
        exit(EXIT_FAILURE);
    }
    rV = rVtmp;
    free(first_vertex);
    first_vertex = (index_t*) malloc_check(sizeof(index_t)*(rV + 1));
    comp_t rv = (comp_t) -1;
    for (index_t v = 0; v < V; v++){
        index_t u = comp_list[v] = get_tmp_comp_list(v);
        if (comp_assign[u] == ASSIGNED_ROOT){ first_vertex[++rv] = v; }
        comp_assign[u] = rv;
    }
    first_vertex[rV] = V;

    return saturation;
}

template <typename real_t, typename index_t, typename comp_t>
void Cp<real_t, index_t, comp_t>::compute_reduced_graph()
/* this could actually be parallelized, but is it worth the pain? */
{
    /* When dealing with component ru, reduced_edge_to[rv] is the number of
     * the reduced edge ru -> rv, or NO_EDGE if the edge is not created yet */
    index_t* reduced_edge_to = (index_t*) malloc_check(sizeof(index_t)*rV);
    for (comp_t rv = 0; rv < rV; rv++){ reduced_edge_to[rv] = NO_EDGE; }

    /* temporary buffer size */
    size_t rEtmp = rE > rV ? rE : rV;

    reduced_edges = (comp_t*) malloc_check(sizeof(comp_t)*2*rEtmp);
    reduced_edge_weights = (real_t*) malloc_check(sizeof(real_t)*rEtmp);

    rE = 0; // current number of reduced edges
    size_t last_rE = 0; // keep track of number of processed edges
    for (comp_t ru = 0; ru < rV; ru++){ /* iterate over the components */
        bool isolated = true; // flag isolated components (useful for PFDR)
        /* run along the component ru */
        for (index_t i = first_vertex[ru]; i < first_vertex[ru + 1]; i++){
            index_t u = comp_list[i];
            for (arc* a = G->nodes[u].first; a; a = a->next){
                if (a->r_cap != ACTIVE_EDGE){ continue; }
                index_t e = (a - G->arcs)/2; // index in undirected edge list
                if (EDGE_WEIGHTS_(e) == ZERO){ continue; }
                isolated = false; // a nonzero edge involving ru exists
                index_t v = a->head - G->nodes; // adjacent vertex
                comp_t rv = comp_assign[v];
                if (rv < ru){ continue; } // count only undirected edges
                index_t re = reduced_edge_to[rv];
                if (re == NO_EDGE){ // a new edge must be created
                    if (rE == rEtmp){ // reach buffer size
                        rEtmp *= 2;
                        reduced_edges = (comp_t*) realloc_check(reduced_edges,
                            sizeof(comp_t)*2*rEtmp);
                        reduced_edge_weights =
                            (real_t*) realloc_check(reduced_edge_weights,
                            sizeof(real_t)*rEtmp);
                    }
                    reduced_edges[2*rE] = ru;
                    reduced_edges[2*rE + 1] = rv;
                    reduced_edge_weights[rE] = EDGE_WEIGHTS_(e);
                    reduced_edge_to[rv] = rE++;
                }else{ /* edge already exists */
                    reduced_edge_weights[re] += EDGE_WEIGHTS_(e);
                }
            }
        }
        if (isolated){ /* this is only useful for solving reduced problems
        * with certain implementation of PFDR where isolated vertices must be
        * linked to themselves */
            if (rE == rEtmp){ // reach buffer size
                rEtmp *= 2;
                reduced_edges = (comp_t*) realloc_check(reduced_edges,
                    sizeof(comp_t)*2*rEtmp);
                reduced_edge_weights = (real_t*) realloc_check(
                    reduced_edge_weights, sizeof(real_t)*rEtmp);
            }
            reduced_edges[2*rE] = reduced_edges[2*rE + 1] = ru;
            reduced_edge_weights[rE++] = eps;
        }else{ /* reset reduced_edge_to */
            for (; last_rE < rE; last_rE++){
                reduced_edge_to[reduced_edges[2*last_rE + 1]] = NO_EDGE;
            }
        }
    }

    free(reduced_edge_to);

    if (rEtmp > rE){
        reduced_edges = (comp_t*) realloc_check(reduced_edges,
            sizeof(comp_t)*2*rE);
        reduced_edge_weights = (real_t*) realloc_check(reduced_edge_weights,
            sizeof(real_t)*rE);
    }
}

template <typename real_t, typename index_t, typename comp_t>
void Cp<real_t, index_t, comp_t>::initialize()
{
    if (!comp_assign){ comp_assign = (comp_t*) malloc_check(sizeof(comp_t)*V); }
    comp_list = (index_t*) malloc_check(sizeof(index_t)*V);

    bool arbitrary = rV == 0;
    /* E/100 is an heuristic ensuring that it is worth parallelizing */
    if (arbitrary){ rV = compute_num_threads((uintmax_t) E/100); }

    if (rV == 1){
        single_connected_component();
        rX = (real_t*) malloc_check(sizeof(real_t)*D);
        solve_univertex_problem(rX);
    }else{
        if (arbitrary){ arbitrary_connected_components(); }
        else{ assign_connected_components(); }
        rX = (real_t*) malloc_check(sizeof(real_t)*D*rV);
        compute_reduced_graph();
        solve_reduced_problem();
        merge();
    }
}

/* by default, relative evolution in Euclidean norm;
 * by default, no check for saturation so the parameter is not used */
template <typename real_t, typename index_t, typename comp_t>
real_t Cp<real_t, index_t, comp_t>::compute_evolution(const bool compute_dif,
    comp_t & saturation)
{
    if (!compute_dif){ return ZERO; }
    real_t dif = ZERO, amp = ZERO;
    #pragma omp parallel for schedule(dynamic) NUM_THREADS(D*V, rV)  \
        reduction(+:dif, amp) 
    for (comp_t rv = 0; rv < rV; rv++){
        real_t* rXv = rX + rv*D;
        real_t rv_dif = 0, rv_amp = 0;
        for (size_t d = 0; d < D; d++){ rv_amp += rXv[d]*rXv[d]; }
        rv_amp *= (first_vertex[rv + 1] - first_vertex[rv]);
        if (is_saturated(rv)){
            real_t* lrXv = last_rX +
                get_tmp_comp_assign(comp_list[first_vertex[rv]])*D;
            for (size_t d = 0; d < D; d++){
                rv_dif += (lrXv[d] - rXv[d])*(lrXv[d] - rXv[d]);
            }
            rv_dif *= (first_vertex[rv + 1] - first_vertex[rv]);
        }else{
            for (index_t v = first_vertex[rv]; v < first_vertex[rv + 1]; v++){
                real_t* lrXv = last_rX + get_tmp_comp_assign(comp_list[v])*D;
                for (size_t d = 0; d < D; d++){
                    rv_dif += (lrXv[d] - rXv[d])*(lrXv[d] - rXv[d]);
                }
            }
        }
        dif += rv_dif;
        amp += rv_amp;
    }
    dif = sqrt(dif);
    amp = sqrt(amp);
    return amp > eps ? dif/amp : dif/eps;
}

/* instantiate for compilation */
template class Cp<float, uint32_t, uint16_t>;

template class Cp<double, uint32_t, uint16_t>;

template class Cp<float, uint32_t, uint32_t>;

template class Cp<double, uint32_t, uint32_t>;
