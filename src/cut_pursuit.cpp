/*=============================================================================
 * Hugo Raguet 2018
 *===========================================================================*/
#include "../include/cut_pursuit.hpp"
#include "../include/omp_num_threads.hpp"

#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)
#define EDGE_WEIGHTS_(e) (edge_weights ? edge_weights[(e)] : homo_edge_weight)
/* avoid overflows */
#define rVp1 ((size_t) rV + 1)
/* specific flags */
#define NOT_ASSIGNED MAX_NUM_COMP
#define ASSIGNED ((comp_t) 1)
#define ASSIGNED_ROOT ((comp_t) 2)
/* maximum number of edges; no edge can have this identifier */
#define NO_EDGE (std::numeric_limits<index_t>::max())

#define TPL template <typename real_t, typename index_t, typename comp_t, \
    typename value_t>
#define CP Cp<real_t, index_t, comp_t, value_t>

using namespace std; 

TPL CP::Cp(index_t V, index_t E, const index_t* first_edge,
    const index_t* adj_vertices, size_t D)
    : V(V), E(E), first_edge(first_edge), adj_vertices(adj_vertices), D(D)
{
    /* construct graph */
    G = new Cp_graph<real_t, index_t, comp_t>(V, E);
    G->add_node(V);
    /* edges */
    for (index_t v = 0; v < V; v++){
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
    reduced_edge_weights = nullptr;
    reduced_edges = nullptr;
    elapsed_time = nullptr;
    objective_values = iterate_evolution = nullptr;
    rX = last_rX = nullptr;
    
    it_max = 10; verbose = 1000;
    dif_tol = ZERO;
    eps = numeric_limits<real_t>::epsilon();
    monitor_evolution = false;
}

TPL CP::~Cp()
{
    delete G;
    free(comp_assign); free(comp_list); free(first_vertex);
    free(reduced_edges); free(reduced_edge_weights);
    free(rX); free(last_rX); 
}

TPL void CP::reset_active_edges()
{ for (index_t e = 0; e < E; e++){ set_inactive(e); } }

TPL void CP::set_edge_weights(const real_t* edge_weights,
    real_t homo_edge_weight)
{
    this->edge_weights = edge_weights;
    this->homo_edge_weight = homo_edge_weight;
}

TPL void CP::set_monitoring_arrays(real_t* objective_values,
    double* elapsed_time, real_t* iterate_evolution)
{
    this->objective_values = objective_values;
    this->elapsed_time = elapsed_time;
    this->iterate_evolution = iterate_evolution;
    if (iterate_evolution){ monitor_evolution = true; }
}

TPL void CP::set_components(comp_t rV, comp_t* comp_assign)
{
    if (rV > 1 && !comp_assign){
        cerr << "Cut-pursuit: if an initial number of components greater than "
            "unity is given, components assignment must be provided." << endl;
        exit(EXIT_FAILURE);
    }
    this->rV = rV;
    this->comp_assign = comp_assign;
}

TPL void CP::set_cp_param(real_t dif_tol, int it_max, int verbose, real_t eps)
{
    this->dif_tol = dif_tol;
    if (dif_tol > ZERO){ monitor_evolution = true; }
    this->it_max = it_max;
    this->verbose = verbose;
    this->eps = ZERO < dif_tol && dif_tol < eps ? dif_tol : eps;
}

TPL comp_t CP::get_components(comp_t** comp_assign, index_t** first_vertex,
    index_t** comp_list)
{
    if (comp_assign){ *comp_assign = this->comp_assign; }
    if (first_vertex){ *first_vertex = this->first_vertex; }
    if (comp_list){ *comp_list = this->comp_list; }
    return this->rV;
}


TPL size_t CP::get_reduced_graph(comp_t** reduced_edges,
    real_t** reduced_edge_weights)
{
    if (reduced_edges){ *reduced_edges = this->reduced_edges; }
    if (reduced_edge_weights){
        *reduced_edge_weights = this->reduced_edge_weights;
    }
    return this->rE;
}

TPL value_t* CP::get_reduced_values(){ return rX; }

TPL void CP::set_reduced_values(value_t* rX){ this->rX = rX; }

TPL int CP::cut_pursuit(bool init)
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
        if (verbose){
            cout << activation << " new activated edge(s)." << endl;
        }

        if (!activation){ /* do not recompute reduced problem */
            saturation = rV;
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
                for (size_t i = 0; i < D*rV; i++){ last_rX[i] = rX[i]; }
                /* store previous assignment */
                for (index_t v = 0; v < V; v++){
                    set_tmp_comp_assign(v, comp_assign[v]);
                }
            }
            free(rX); rX = nullptr;
            free(reduced_edges); reduced_edges = nullptr;
            free(reduced_edge_weights); reduced_edge_weights = nullptr;
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
        if (verbose){
            cout << deactivation << " deactivated edge(s)." << endl;
        }

        if (monitor_evolution){
            dif = compute_evolution(dif_tol > ZERO || iterate_evolution,
                saturation);
            if (iterate_evolution){ iterate_evolution[it] = dif; }
            free(last_rX); last_rX = nullptr;
        }

        it++;

        if (objective_values){ objective_values[it] = compute_objective(); }
    } /* endwhile true */

    return it;
}

TPL double CP::monitor_time(chrono::steady_clock::time_point start)
{ 
    using namespace chrono;
    steady_clock::time_point current = steady_clock::now();
    return ((current - start).count()) * steady_clock::period::num
               / static_cast<double>(steady_clock::period::den);
}

TPL void CP::print_progress(int it, real_t dif, comp_t saturation,
    double timer)
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

TPL void CP::single_connected_component()
{
    for (index_t v = 0; v < V; v++){ comp_assign[v] = 0; }
    free(first_vertex);
    first_vertex = (index_t*) malloc_check(sizeof(index_t)*2);
    first_vertex[0] = 0; first_vertex[1] = V;
    for (index_t v = 0; v < V; v++){ comp_list[v] = v; }
    set_saturation(0, false);
}

TPL void CP::new_connected_component(comp_t& rv, index_t& comp_size)
{
    if (rv == rV){
        rV = (size_t) 2*rV < MAX_NUM_COMP ? (size_t) 2*rV : MAX_NUM_COMP;
        first_vertex = (index_t*) realloc_check(first_vertex,
            sizeof(index_t)*rVp1);
    }
    first_vertex[rv + 1] = first_vertex[rv] + comp_size;
    set_saturation(rv, false);
    rv++; comp_size = 0;
    if (rv == MAX_NUM_COMP){
        cerr << "Cut-pursuit: number of components greater than can be "
            "represented by comp_t (" << MAX_NUM_COMP << ")." << endl;
        exit(EXIT_FAILURE);
    }
}

TPL void CP::arbitrary_connected_components()
{
    index_t max_comp_size = V/rV + 1;
    free(first_vertex);
    first_vertex = (index_t*) malloc_check(sizeof(index_t)*rVp1);

    /* cleanup assigned components */
    for (index_t v = 0; v < V; v++){ comp_assign[v] = NOT_ASSIGNED; }

    index_t comp_size = 0;
    comp_t rv = 0; // identify and count components
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
                    new_connected_component(rv, comp_size);
                }
            }
        } // the current connected component cannot grow anymore
        if (comp_size > 0){ // add the current component
            new_connected_component(rv, comp_size);
        }
    }
    /* update components lists and assignments */
    if (rV > rv){
        rV = rv;
        first_vertex = (index_t*) realloc_check(first_vertex,
            sizeof(index_t)*rVp1);
    }
}

TPL void CP::assign_connected_components()
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
    free(first_vertex);
    first_vertex = (index_t*) malloc_check(sizeof(index_t)*rVp1);
    for (comp_t rv = 0; rv < rVp1; rv++){ first_vertex[rv] = 0; }
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

TPL comp_t CP::compute_connected_components()
{
    comp_t saturation = 0;

    /* cleanup assigned components */
    for (index_t v = 0; v < V; v++){ comp_assign[v] = NOT_ASSIGNED; }

    index_t rVtmp = 0; // identify and count components, prevent overflow
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

    if (rVtmp > MAX_NUM_COMP){
        cerr << "Cut-pursuit: number of components (" << rVtmp << ") greater "
            << "than can be represented by comp_t (" << MAX_NUM_COMP << ")"
            << endl;
        exit(EXIT_FAILURE);
    }

    /* update components lists and assignments */
    rV = rVtmp;
    free(first_vertex);
    first_vertex = (index_t*) malloc_check(sizeof(index_t)*rVp1);
    comp_t rv = (comp_t) -1;
    for (index_t i = 0; i < V; i++){
        index_t v = comp_list[i] = get_tmp_comp_list(i);
        if (comp_assign[v] == ASSIGNED_ROOT){ first_vertex[++rv] = i; }
        comp_assign[v] = rv;
    }
    first_vertex[rV] = V;

    return saturation;
}

TPL void CP::compute_reduced_graph()
/* this could actually be parallelized, but is it worth the pain? */
{
    free(reduced_edges);
    free(reduced_edge_weights);

    if (rV == 1){ /* reduced graph only edge from the component to itself
                   * this is only useful for solving reduced problems with
                   * certain implementations where isolated vertices must be
                   * linked to themselves */
        rE = 1;
        reduced_edges = (comp_t*) malloc_check(sizeof(comp_t)*2);
        reduced_edges[0] = reduced_edges[1] = 0;
        reduced_edge_weights = (real_t*) malloc_check(sizeof(real_t)*1);
        reduced_edge_weights[0] = eps;
        return; 
    }

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
        bool isolated = true; // flag isolated components
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
                        rEtmp += rEtmp;
                        reduced_edges = (comp_t*) realloc_check(reduced_edges,
                            sizeof(comp_t)*2*rEtmp);
                        reduced_edge_weights = (real_t*) realloc_check(
                            reduced_edge_weights, sizeof(real_t)*rEtmp);
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
        * with certain implementations where isolated vertices must be linked
        * to themselves */
            if (rE == rEtmp){ // reach buffer size
                rEtmp += rEtmp;
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

TPL void CP::initialize()
{
    if (!comp_assign){
        comp_assign = (comp_t*) malloc_check(sizeof(comp_t)*V);
    }
    if (!comp_list){
        comp_list = (index_t*) malloc_check(sizeof(index_t)*V);
    }

    bool arbitrary = rV == 0;
    /* E/100 is an heuristic ensuring that it is worth parallelizing */
    if (arbitrary){ rV = compute_num_threads((uintmax_t) E/100); }

    if (rV == 1){
        single_connected_component();
    }else{
        if (arbitrary){ arbitrary_connected_components(); }
        else{ assign_connected_components(); }
    }

    compute_reduced_graph();
    solve_reduced_problem();
    merge();
}

TPL void CP::merge_components(comp_t& ru, comp_t& rv)
{
    /* ensure the smallest component will be the root of the merge chain */
    if (ru > rv){ comp_t tmp = ru; ru = rv; rv = tmp; }
    /* link both chains; update leaf of the merge chain; update root info */
    merge_chains_next[merge_chains_leaf[ru]] = rv;
    merge_chains_leaf[ru] = merge_chains_leaf[rv];
    merge_chains_root[rv] = merge_chains_root[merge_chains_leaf[rv]] = ru;
    set_saturation(ru, false); // component has changed
}

TPL index_t CP::merge()
{
    /* if (rE == 0){ return 0; } currently, isolated compoents are linked
     * to themselve in the reduced graph, thus rE is at least one */
    
    /* create the chains representing the merged components */
    merge_chains_root = (comp_t*) malloc_check(sizeof(comp_t)*rV); 
    merge_chains_next = (comp_t*) malloc_check(sizeof(comp_t)*rV);
    merge_chains_leaf = (comp_t*) malloc_check(sizeof(comp_t)*rV);
    for (comp_t rv = 0; rv < rV; rv++){
        merge_chains_root[rv] = CHAIN_ROOT;
        merge_chains_next[rv] = CHAIN_LEAF;
        merge_chains_leaf[rv] = rv;
    }
    comp_t merge_count = compute_merge_chains();
    if (!merge_count){
        free(merge_chains_root);
        free(merge_chains_next);
        free(merge_chains_leaf);
        return 0;
    }

    /* construct a new version of 'comp_list' in temporary storage and update
     * 'rX', 'first_vertex', and 'reduced_edges' and '_weights' in-place */
    comp_t* new_comp_id = merge_chains_leaf; // reuse storage
    comp_t rn = 0; // component number
    index_t i = 0; // index in the new comp_list
    for (comp_t ru = 0; ru < rV; ru++){
        if (merge_chains_root[ru] != CHAIN_ROOT){ continue; }
        /* otherwise ru is a root, create the corresponding new component */
        const value_t* rXu = rX + D*ru;
        value_t* rXn = rX + D*rn;
        for (size_t d = 0; d < D; d++){ rXn[d] = rXu[d]; }
        comp_t rv = ru; 
        index_t first = i; // holds index of first vertex of the component
        while (rv != CHAIN_LEAF){
            new_comp_id[rv] = rn;
            for (index_t v = first_vertex[rv]; v < first_vertex[rv + 1]; v++){
                set_tmp_comp_list(i++, comp_list[v]);
            }
            rv = merge_chains_next[rv];
        }
        /* the root of each chain is the smallest component in the chain, so
         * now that 'rn' new components have been constructed, the first 'rn'
         * previous components have been copied, hence 'first_vertex' will not
         * be accessed before position 'rn' anymore; thus modify in-place */
        first_vertex[rn++] = first;
    }
    /* finalize and shrink arrays to fit the reduced number of components */
    first_vertex[rV = rn] = V;
    first_vertex = (index_t*) realloc_check(first_vertex,
        sizeof(index_t)*rVp1);
    rX = (real_t*) realloc_check(rX, sizeof(real_t)*D*rV);
    for (index_t v = 0; v < V; v++){ /* update components assignments */
        comp_list[v] = get_tmp_comp_list(v);
        comp_assign[v] = new_comp_id[comp_assign[v]];
    }

    /* update corresponding reduced edges;
     * some edges will appear several times in the list, important thing is
     * that the corresponding weights sum up to the right quantity;
     * note that rE is thus an upper bound of the actual number of edges */
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
    rE = new_re;
    reduced_edges = (comp_t*) realloc_check(reduced_edges,
            sizeof(comp_t)*2*rE);
    reduced_edge_weights = (real_t*) realloc_check(reduced_edge_weights,
            sizeof(real_t)*rE);

    free(merge_chains_root);
    free(merge_chains_next);
    free(merge_chains_leaf);

    /* deactivate corresponding edges */
    index_t deactivation = 0;
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

    return deactivation;
}

/* instantiate for compilation */
template class Cp<float, uint32_t, uint16_t>;
template class Cp<double, uint32_t, uint16_t>;
template class Cp<float, uint32_t, uint32_t>;
template class Cp<double, uint32_t, uint32_t>;
