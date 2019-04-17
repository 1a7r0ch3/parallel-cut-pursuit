/*=============================================================================
 * Hugo Raguet 2018
 *===========================================================================*/
#include <cmath>
#include "../include/cp_pfdr_d1_ql1b.hpp"
#include "../include/omp_num_threads.hpp"
#include "../include/matrix_tools.hpp"
#include "../include/pfdr_d1_ql1b.hpp"
#include "../include/wth_element.hpp"

#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)
#define HALF ((real_t) 0.5)
#define EDGE_WEIGHTS_(e) (edge_weights ? edge_weights[(e)] : homo_edge_weight)
#define L1_WEIGHTS_(v) (l1_weights ? l1_weights[(v)] : homo_l1_weight)
#define Y_(n) (Y ? Y[(n)] : (real_t) 0.0)
#define Yl1_(v) (Yl1 ? Yl1[(v)] : (real_t) 0.0)

#define TPL template <typename real_t, typename index_t, typename comp_t>
#define CP_D1_QL1B Cp_d1_ql1b<real_t, index_t, comp_t>

using namespace std;

TPL CP_D1_QL1B::Cp_d1_ql1b(index_t V, index_t E, const index_t* first_edge,
    const index_t* adj_vertices)
    : Cp_d1<real_t, index_t, comp_t>(V, E, first_edge, adj_vertices)
{
    /* ensure handling of infinite values (negation, comparisons) is safe */
    static_assert(numeric_limits<real_t>::is_iec559,
        "Cut-pursuit d1 quadratic l1 bounds: real_t must satisfy IEEE 754.");
    Y = Yl1 = A = R = nullptr;
    N = DIAG_ATA;
    a = ONE;
    l1_weights = nullptr; homo_l1_weight = ZERO;
    low_bnd = nullptr; homo_low_bnd = -INF_REAL;
    upp_bnd = nullptr; homo_upp_bnd = INF_REAL;

    pfdr_rho = 1.0; pfdr_cond_min = 1e-3; pfdr_dif_rcd = 0.0;
    pfdr_dif_tol = 1e-3*dif_tol; pfdr_it = pfdr_it_max = 1e4;

    /* it makes sense to consider nonevolving components as saturated;
     * beware of coupling when using complicated operator A though,
     * precision can be increased by decreasing dif_tol if necessary */
    monitor_evolution = true;
}

TPL CP_D1_QL1B::~Cp_d1_ql1b(){ free(R); }

TPL void CP_D1_QL1B::set_quadratic(const real_t* Y, size_t N, const real_t* A,
    real_t a)
{
    if (!A && a == ZERO){ // no quadratic part !
        N = DIAG_ATA;
    }
    free(R);
    R = IS_ATA(N) ? nullptr : (real_t*) malloc_check(sizeof(real_t)*N);
    this->Y = Y; this->N = N; this->A = A; this->a = a;
}

TPL void CP_D1_QL1B::set_l1(const real_t* l1_weights, real_t homo_l1_weight,
    const real_t* Yl1)
{
    if (!l1_weights && homo_l1_weight < ZERO){
        cerr << "Cut-pursuit graph d1 quadratic l1 bounds: negative "
            "homogeneous l1 penalization (" << homo_l1_weight << ")." << endl;
        exit(EXIT_FAILURE);
    }
    this->l1_weights = l1_weights; this->homo_l1_weight = homo_l1_weight;
    this->Yl1 = Yl1;
}

TPL void CP_D1_QL1B::set_bounds(const real_t* low_bnd, real_t homo_low_bnd,
    const real_t* upp_bnd, real_t homo_upp_bnd)
{
    if (!low_bnd && !upp_bnd && homo_low_bnd > homo_upp_bnd){
        cerr << "Cut-pursuit graph d1 quadratic l1 bounds: homogeneous lower "
            "bound (" << homo_low_bnd << ") greater than homogeneous upper "
            "bound (" << homo_upp_bnd << ")." << endl;
        exit(EXIT_FAILURE);
    }
    this->low_bnd = low_bnd; this->homo_low_bnd = homo_low_bnd;
    this->upp_bnd = upp_bnd; this->homo_upp_bnd = homo_upp_bnd;
}

TPL void CP_D1_QL1B::set_pfdr_param(real_t rho, real_t cond_min,
    real_t dif_rcd, int it_max, real_t dif_tol)
{
    this->pfdr_rho = rho;
    this->pfdr_cond_min = cond_min;
    this->pfdr_dif_rcd = dif_rcd;
    this->pfdr_it_max = it_max;
    this->pfdr_dif_tol = dif_tol;
}

TPL void CP_D1_QL1B::solve_reduced_problem()
/* NOTA: if Yl1 is not constant, this solves only an approximation, replacing
 * the weighted sum of distances to Yl1 by the distance to the weighted median
 * of Yl1 */
{
    /**  compute reduced matrix  **/
    real_t *rY, *rA, *rAA; // reduced observations, matrix, etc.
    rY = rA = rAA = nullptr;
    /* rN conveys information on the matricial shape; even if the main problem
     * uses a direct matricial form (indicated by positive N), one might still
     * uses premultiplication for the reduced problem; rule of thumb to decide:
     * without premultiplication: 2 N rV i operations
     *     + two matrix-vector mult. per PFDR iter. : 2 N rV i
     * with premultiplication: N rV^2 + rV^2 i operations
     *     + compute symmetrized reduced matrix: N rV^2
     *     + one matrix-vector mult. per pfdr iter. : rV^2 i
     * conclusion: premultiplication if rV < (2 N i)/(N + i) */
    size_t rN = !IS_ATA(N) && rV < (2*N*pfdr_it)/(N + pfdr_it) ? FULL_ATA : N;

    if (IS_ATA(rN)){ /* reduced problem premultiplied by rA^t */
        if (Y){ rY = (real_t*) malloc_check(sizeof(real_t)*rV); }
        if (A || a){
            if (N == DIAG_ATA){
                rAA = (real_t*) malloc_check(sizeof(real_t)*rV);
            }else{ // FULL_ATA or direct matricial case with premultiplication
                rAA = (real_t*) malloc_check(sizeof(real_t)*rV*rV);
                rN = FULL_ATA;
            }
        }
    }
    if (!IS_ATA(N)){ /* direct matricial main problem */
        rA = (real_t*) malloc_check(sizeof(real_t)*N*rV);
        for (size_t i = 0; i < N*rV; i++){ rA[i] = ZERO; }
        #pragma omp parallel for schedule(dynamic) NUM_THREADS(N*V, rV)
        for (comp_t rv = 0; rv < rV; rv++){
            real_t *rAv = rA + N*rv; // rv-th column of rA
            /* run along the component rv */
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                const real_t *Av = A + N*comp_list[i];
                for (size_t n = 0; n < N; n++){ rAv[n] += Av[n]; }
            }
        }
        if (rN == FULL_ATA){
            /* fill upper triangular part of rA^t rA */
            #pragma omp parallel for schedule(dynamic) \
                NUM_THREADS(N*rV*rV/2, rV)
            for (comp_t ru = 0; ru < rV; ru++){
                real_t *rAu = rA + N*ru; // ru-th column of rA
                real_t *rAAu = rAA + (size_t) rV*ru; // ru-th column of rAA
                for (comp_t rv = 0; rv <= ru; rv++){
                    real_t *rAv = rA + N*rv; // rv-th column of rA
                    rAAu[rv] = ZERO;
                    for (size_t n = 0; n < N; n++){
                        rAAu[rv] += rAu[n]*rAv[n];
                    }
                }
            }
            if (Y){ /* correlation with observation Y */
                #pragma omp parallel for schedule(static) NUM_THREADS(rV*N, rV)
                for (comp_t rv = 0; rv < rV; rv++){
                    rY[rv] = ZERO;
                    real_t *rAv = rA + N*rv; // rv-th column of rA
                    for (size_t n = 0; n < N; n++){ rY[rv] += rAv[n]*Y[n]; }
                }
            }
            /* keep also rA for later update of the residual */
        }
    }else{ /* main problem premultiplied by A^t */
        if (Y){ /* recall that observation Y is actually A^t Y */
            #pragma omp parallel for schedule(dynamic) NUM_THREADS(V, rV)
            for (comp_t rv = 0; rv < rV; rv++){
                rY[rv] = ZERO;
                /* run along the component rv */
                for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1];
                    i++){
                    rY[rv] += Y[comp_list[i]];
                }
            }
        }
        if (N == FULL_ATA){ /* full matrix */
            /* fill upper triangular part of rA^t rA */
            #pragma omp parallel for schedule(dynamic) NUM_THREADS(V*V/2, rV)
            for (comp_t ru = 0; ru < rV; ru++){
                real_t* rAAu = rAA + (size_t) rV*ru;
                for (comp_t rv = 0; rv <= ru; rv++){
                    rAAu[rv] = ZERO;
                    /* run along the component ru */
                    for (index_t i = first_vertex[ru];
                        i < first_vertex[ru + 1]; i++){
                        const real_t *Au = A + (size_t) V*comp_list[i];
                        /* run along the component rv */
                        for (index_t j = first_vertex[rv];
                                j < first_vertex[rv + 1]; j++){
                            rAAu[rv] += Au[comp_list[j]];
                        }
                    }
                }
            }
        }else if (A){ /* diagonal matrix */
            #pragma omp parallel for schedule(dynamic) NUM_THREADS(V, rV)
            for (comp_t rv = 0; rv < rV; rv++){
                rAA[rv] = ZERO;
                /* run along the component rv */
                for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1];
                    i++){
                    rAA[rv] += A[comp_list[i]];
                }
            }
        }else if (a){ /* identity */
            #pragma omp parallel for schedule(static) NUM_THREADS(rV)
            for (comp_t rv = 0; rv < rV; rv++){
                rAA[rv] = first_vertex[rv + 1] - first_vertex[rv];
            }
        }
    }
    if (rN == FULL_ATA){ /* fill lower triangular part of rA^t rA */
        #pragma omp parallel for schedule(dynamic) NUM_THREADS(rV*rV/2, rV)
        for (comp_t ru = 0; ru < rV - 1; ru++){
            real_t *rAAu = rAA + (size_t) rV*ru;
            size_t i = rV + (size_t) (rV + 1)*ru;
            for (comp_t rv = ru + 1; rv < rV; rv++){
                rAAu[rv] = rAA[i];
                i += rV;
            }
        }
    }

    /**  compute reduced l1 weights, medians and bounds  **/
    real_t *rl1_weights, *rYl1, *rlow_bnd, *rupp_bnd;
    rl1_weights = rYl1 = rlow_bnd = rupp_bnd = nullptr;
    uintmax_t num_ops = 0;
    if (l1_weights || homo_l1_weight){
        rl1_weights = (real_t*) malloc_check(sizeof(real_t)*rV);
        num_ops += l1_weights ? V : rV;
    }
    if (Yl1){
        rYl1 = (real_t*) malloc_check(sizeof(real_t)*rV);
        num_ops += V;
    }
    if (low_bnd){
        rlow_bnd = (real_t*) malloc_check(sizeof(real_t)*rV);
        num_ops += V;
    }
    if (upp_bnd){
        rupp_bnd = (real_t*) malloc_check(sizeof(real_t)*rV);
        num_ops += V;
    }
    if (num_ops){
        #pragma omp parallel for schedule(dynamic) NUM_THREADS(num_ops, rV)
        for (comp_t rv = 0; rv < rV; rv++){
            if (l1_weights){
                rl1_weights[rv] = ZERO;
                /* run along the component rv */
                for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1];
                    i++){
                    rl1_weights[rv] += l1_weights[comp_list[i]];
                }
                if (Yl1){
                    /* saturation is flagged on first vertex */
                    bool saturation = is_saturated(rv);
                    rYl1[rv] = wth_element(comp_list + first_vertex[rv],
                        Yl1, first_vertex[rv + 1] - first_vertex[rv],
                        (double) HALF*rl1_weights[rv], l1_weights);
                    /* ordering has changed, retrieve saturation info */
                    set_saturation(rv, saturation);
                }
            }else if (homo_l1_weight){
                rl1_weights[rv] = (first_vertex[rv + 1] - first_vertex[rv])
                    *homo_l1_weight;
                if (Yl1){
                    /* saturation is flagged on first vertex */
                    bool saturation = is_saturated(rv);
                    rYl1[rv] = nth_element_idx(comp_list + first_vertex[rv],
                        Yl1, first_vertex[rv + 1] - first_vertex[rv],
                        (first_vertex[rv + 1] - first_vertex[rv])/2);
                    /* ordering has changed, retrieve saturation info */
                    set_saturation(rv, saturation);
                }
            }
            if (low_bnd){
                rlow_bnd[rv] = -INF_REAL;
                /* run along the component rv */
                for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1];
                    i++){
                    if (rlow_bnd[rv] < low_bnd[comp_list[i]]){
                        rlow_bnd[rv] = low_bnd[comp_list[i]];
                    }
                }
            }
            real_t *rupp_bnd = nullptr;
            if (upp_bnd){
                rupp_bnd[rv] = INF_REAL;
                /* run along the component rv */
                for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1];
                    i++){
                    if (rupp_bnd[rv] > upp_bnd[comp_list[i]]){
                        rupp_bnd[rv] = upp_bnd[comp_list[i]];
                    }
                }
            }
        }
    }

    if (rV == 1){ /**  single connected component  **/

        /* solution of least-square + l1 */
        real_t wl1 = rl1_weights ? *rl1_weights : ZERO;
        real_t yl1 = rYl1 ? *rYl1 : ZERO;
        if (*rY - wl1 > (*rAA)*yl1){ *rX = (*rY - wl1)/(*rAA); }
        else if (*rY + wl1 < (*rAA)*yl1){ *rX = (*rY + wl1)/(*rAA); }
        else{ *rX = yl1; }

        /* aggregated lower bounds and proj */
        real_t low = low_bnd ? *rlow_bnd : homo_low_bnd;
        real_t upp = upp_bnd ? *rupp_bnd : homo_upp_bnd;
        if (*rX < low){ *rX = low; }
        if (*rX > upp){ *rX = upp; }

    }else{ /**  preconditioned forward-Douglas-Rachford  **/

        Pfdr_d1_ql1b<real_t, comp_t> *pfdr =
            new Pfdr_d1_ql1b<real_t, comp_t>(rV, rE, reduced_edges);

        pfdr->set_edge_weights(reduced_edge_weights);
        if (IS_ATA(rN)){ pfdr->set_quadratic(rY, rN, rAA, a); }
        else{ pfdr->set_quadratic(Y, N, rA); }
        pfdr->set_l1(rl1_weights, ZERO, rYl1);
        pfdr->set_bounds(rlow_bnd, homo_low_bnd, rupp_bnd, homo_upp_bnd);
        pfdr->set_conditioning_param(pfdr_cond_min, pfdr_dif_rcd);
        pfdr->set_relaxation(pfdr_rho);
        pfdr->set_algo_param(pfdr_dif_tol, pfdr_it_max, verbose);
        pfdr->set_iterate(rX);
        pfdr->initialize_iterate();

        pfdr_it = pfdr->precond_proximal_splitting();

        pfdr->set_iterate(nullptr); // prevent rX to be free()'d
        delete pfdr;

    }

    if (!IS_ATA(N)){ /* direct matricial case, compute residual R = Y - A X */
        #pragma omp parallel for schedule(static) NUM_THREADS(N*rV, N)
        for (size_t n = 0; n < N; n++){
            R[n] = Y_(n);
            size_t i = n;
            for (comp_t rv = 0; rv < rV; rv++){
                R[n] -= rA[i]*rX[rv];
                i += N;
            }
        }
    }

    free(rY); free(rA); free(rAA); free(rYl1);
    free(rl1_weights); free(rlow_bnd); free(rupp_bnd);
}

TPL index_t CP_D1_QL1B::split()
{
    index_t activation = 0;
    real_t* grad = (real_t*) malloc_check(sizeof(real_t)*V);
    for (index_t v = 0; v < V; v++){ grad[v] = ZERO; }

    /**  gradient of quadratic term  **/ 
    if (!IS_ATA(N)){ /* direct matricial case, grad = -(A^t) R */
        #pragma omp parallel for schedule(static) NUM_THREADS(V*N, V)
        for (index_t v = 0; v < V; v++){
            const real_t *Av = A + N*v;
            for (size_t n = 0; n < N; n++){ grad[v] -= Av[n]*R[n]; }
        }
    }else if (N == FULL_ATA){ /* grad = (A^t A)*X - A^t Y  */
        #pragma omp parallel for schedule(static) NUM_THREADS(V*V, V)
        for (index_t u = 0; u < V; u++){
            const real_t *Au = A + (size_t) V*u;
            for (comp_t rv = 0; rv < rV; rv++){
                if (rX[rv] == ZERO){ continue; }
                real_t aurv = ZERO; /* sum u-th row of (A^t A), rv-th comp */
                /* run along the component rv */
                for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1];
                    i++){
                    /* can sum column wise, by symmetry */
                    aurv += Au[comp_list[i]]; 
                }
                grad[u] += aurv*rX[rv];
            }
            grad[u] -= Y_(u);
        }
    }else if (A){ /* diagonal case, grad = (A^t A) X - A^t Y */
        #pragma omp parallel for schedule(static) NUM_THREADS(V)
        for (index_t v = 0; v < V; v++){
            grad[v] = A[v]*rX[comp_assign[v]] - Y_(v);
        }
    }else if (a){ /* identity matrix */
        #pragma omp parallel for schedule(static) NUM_THREADS(V)
        for (index_t v = 0; v < V; v++){
            grad[v] = rX[comp_assign[v]] - Y_(v);
        }
    }

    /**  differentiable d1 contribution to the gradient  **/ 
    #pragma omp parallel for schedule(static) NUM_THREADS(E, V)
    for (index_t u = 0; u < V; u++){
        for (index_t e = first_edge[u]; e < first_edge[u + 1]; e++){
            if (is_active(e)){
                index_t v = adj_vertices[e];
                real_t grad_d1 = rX[comp_assign[u]] > rX[comp_assign[v]] ?
                    EDGE_WEIGHTS_(e) : -EDGE_WEIGHTS_(e);
                grad[u] += grad_d1;
                grad[v] -= grad_d1;
            }
        }
    }

    /**  differentiable l1 contribution  **/
    if (l1_weights || homo_l1_weight){
        #pragma omp parallel for schedule(dynamic) NUM_THREADS(V, rV)
        for (comp_t rv = 0; rv < rV; rv++){
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                index_t v = comp_list[i];
                if (rX[rv] > Yl1_(v)){ grad[v] += L1_WEIGHTS_(v); }
                else if (rX[rv] < Yl1_(v)){ grad[v] -= L1_WEIGHTS_(v); }
            }
        }
    }

    /**  set capacities and compute min cuts in parallel along components  **/
    #pragma omp parallel NUM_THREADS(2*V + 5*E, rV)
    {

    Cp_graph<real_t, index_t, comp_t>* Gpar = get_parallel_flow_graph();

    #pragma omp for schedule(dynamic) reduction(+:activation)
    for (comp_t rv = 0; rv < rV; rv++){
        if (is_saturated(rv)){ continue; }
        index_t rv_activation = 0;

        /**  first cut: directions +1_U  **/

        /* set the source/sink capacities */
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            set_term_capacities(v, grad[v]);
        }
        /* l1 contribution is positive at zero */
        if (l1_weights || homo_l1_weight){ 
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                index_t v = comp_list[i];
                if (rX[rv] == Yl1_(v)){
                    add_term_capacities(v, L1_WEIGHTS_(v));
                }
            }
        }
        /* box constraint contribution is infinite at the upper bound */
        if (upp_bnd){
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                index_t v = comp_list[i];
                if (rX[rv] == upp_bnd[v]){ set_term_capacities(v, INF_REAL); }
            }
        }else if (homo_upp_bnd < INF_REAL && rX[rv] == homo_upp_bnd){
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                set_term_capacities(comp_list[i], INF_REAL);
            }
        }
        /* set the d1 edge capacities */
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
                if (!is_active(e)){
                    set_edge_capacities(e, EDGE_WEIGHTS_(e), EDGE_WEIGHTS_(e));
                }
            }
        }
        /* find min cut and activate edges correspondingly */
        Gpar->maxflow(first_vertex[rv + 1] - first_vertex[rv],
            comp_list + first_vertex[rv]);

        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
                if (!is_active(e) && is_sink(v) != is_sink(adj_vertices[e])){
                    set_active(e);
                    rv_activation++;
                }
            }
        }

        /**  when no nondifferentiable part exists besides the total variation, 
         **  only one cut is required, for direction 1_U - 1_Uc, and it is 
         **  equivalent to the above cut  **/
        if (!l1_weights && !homo_l1_weight && !low_bnd && !upp_bnd &&
            homo_low_bnd == -INF_REAL && homo_upp_bnd == INF_REAL){
            set_saturation(rv, rv_activation == 0);
            activation += rv_activation;
            continue;
        }

        /**  second cut: directions -1_U  **/
        /* set the source/sink capacities */
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            set_term_capacities(v, grad[v]);
        }
        /* l1 contribution is negative at zero */
        if (l1_weights || homo_l1_weight){
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                index_t v = comp_list[i];
                if (rX[rv] == Yl1_(v)){
                    add_term_capacities(v, -L1_WEIGHTS_(v));
                }
            }
        }
        /* box constraint contribution is infinite at the lower bound */
        if (low_bnd){
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                index_t v = comp_list[i];
                if (rX[rv] == low_bnd[v]){ set_term_capacities(v, -INF_REAL); }
            }
        }else if (homo_low_bnd > -INF_REAL && rX[rv] == homo_low_bnd){
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                set_term_capacities(comp_list[i], -INF_REAL);
            }
        }
        /* set the d1 edge capacities */
        #pragma omp parallel for schedule(static) NUM_THREADS(E)
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
                if (!is_active(e)){
                    set_edge_capacities(e, EDGE_WEIGHTS_(e), EDGE_WEIGHTS_(e));
                }
            }
        }
        /* find min cut and activate edges correspondingly */
        Gpar->maxflow(first_vertex[rv + 1] - first_vertex[rv],
            comp_list + first_vertex[rv]);

        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
                if (!is_active(e) && is_sink(v) != is_sink(adj_vertices[e])){
                    set_active(e);
                    rv_activation++;
                }
            }
        }

        set_saturation(rv, rv_activation == 0);
        activation += rv_activation;

    } // end for rv

    delete Gpar;

    } // end parallel region

    free(grad);
    return activation;
}

TPL real_t CP_D1_QL1B::compute_evolution(bool compute_dif)
{
    size_t num_ops = compute_dif ? V : saturation_count;
    real_t dif = ZERO, amp = ZERO;
    comp_t saturation_par_count = 0; // auxiliary variable for parallel region
    #pragma omp parallel for schedule(dynamic) NUM_THREADS(num_ops, rV) \
        reduction(+:dif, amp, saturation_par_count)
    for (comp_t rv = 0; rv < rV; rv++){  
        real_t rXv = rX[rv];
        if (is_saturated(rv)){
            real_t lrXv =
                last_rX[get_tmp_comp_assign(comp_list[first_vertex[rv]])];
            real_t rv_dif = abs(rXv - lrXv);
            if (rv_dif > abs(rX[rv])*dif_tol){ set_saturation(rv, false); }
            else{ saturation_par_count++; }
            if (compute_dif){
                dif += rv_dif*rv_dif*(first_vertex[rv + 1] - first_vertex[rv]);
                amp += rXv*rXv*(first_vertex[rv + 1] - first_vertex[rv]);
            }
        }else if (compute_dif){
            for (index_t v = first_vertex[rv]; v < first_vertex[rv + 1]; v++){
                real_t lrXv = last_rX[get_tmp_comp_assign(comp_list[v])];
                dif += (rXv - lrXv)*(rXv - lrXv);
            }
            amp += rXv*rXv*(first_vertex[rv + 1] - first_vertex[rv]);
        }
    }
    saturation_count = saturation_par_count;
    if (compute_dif){
        dif = sqrt(dif);
        amp = sqrt(amp);
        return amp > eps ? dif/amp : dif/eps;
    }else{
        return INF_REAL;
    }
}

TPL real_t CP_D1_QL1B::compute_objective()
/* unfortunately, at this point one does not have access to the reduced objects
 * computed in the routine solve_reduced_problem() */
{
    real_t obj = ZERO;

    /* quadratic term */
    if (!IS_ATA(N)){ /* direct matricial case, 1/2 ||Y - A X||^2 */
        #pragma omp parallel for reduction(+:obj) schedule(static) \
            NUM_THREADS(N)
        for (size_t n = 0; n < N; n++){ obj += R[n]*R[n]; }
        obj *= HALF;
    /* premultiplied by A^t, 1/2 <X, A^t A X> - <X, A^t Y> */
    }else if (N == FULL_ATA){ /* full matrix */
        #pragma omp parallel for reduction(+:obj) schedule(dynamic) \
            NUM_THREADS(V*V/2, rV)
        for (comp_t ru = 0; ru < rV; ru++){
            /* 1/2 <X, A^t A X> = 1/2 <rX, rA^t rA rX> = sum{ru} rXu *
             *  ( sum{rv < ru} (rA^t rA)uv rXv + 1/2 (rA^t rA)uu rXu ) */
            real_t sumrAAuvXv = ZERO;
            for (comp_t rv = 0; rv <= ru; rv++){
                real_t rAAuv = ZERO;
                /* run along the component ru */
                for (index_t i = first_vertex[ru]; i < first_vertex[ru + 1];
                    i++){
                    const real_t *Au = A + (size_t) V*comp_list[i];
                    /* run along the component rv */
                    for (index_t j = first_vertex[rv];
                        j < first_vertex[rv + 1]; j++){
                        rAAuv += Au[comp_list[j]];
                    }
                }
                if (rv < ru){ sumrAAuvXv += rAAuv*rX[rv]; }
                else{ sumrAAuvXv += HALF*rAAuv*rX[ru]; }
            }
            real_t rAYu = ZERO;
            for (index_t i = first_vertex[ru]; i < first_vertex[ru + 1];
                i++){
                /* observation Y is actually A^t Y */
                rAYu += Y_(comp_list[i]);
            }
            obj += rX[ru]*(sumrAAuvXv - rAYu);
        }
    }else if (A || a){ /* diagonal matrix */
        #pragma omp parallel for reduction(+:obj) schedule(dynamic) \
            NUM_THREADS(V, rV)
        for (comp_t rv = 0; rv < rV; rv++){
            real_t rAAv = A ? ZERO : // arbitrary diagonal matrix
                first_vertex[rv + 1] - first_vertex[rv]; // identity
            real_t rAYv = ZERO;
            /* run along the component rv */
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                if (A){ rAAv += A[comp_list[i]]; }
                /* observation Y is actually A^t Y */
                rAYv += Y_(comp_list[i]);
            }
            obj += rX[rv]*(HALF*rAAv*rX[rv] - rAYv);
        }
    }

    obj += compute_graph_d1(); // ||x||_d1

    /* ||x||_l1 */
    if (l1_weights){ /* ||x||_l1 */
        #pragma omp parallel for schedule(static) NUM_THREADS(V) \
             reduction(+:obj)
        for (index_t v = 0; v < V; v++){
            obj += l1_weights[v]*abs(rX[comp_assign[v]] - Yl1_(v));
        }
    }else if (homo_l1_weight){
        real_t l1 = ZERO;
        /* run along the component rv */
        #pragma omp parallel for schedule(dynamic) NUM_THREADS(V, rV) \
             reduction(+:l1)
        for (comp_t rv = 0; rv < rV; rv++){
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                l1 += abs(rX[rv] - Yl1_(comp_list[i]));
            }
        }
        obj += homo_l1_weight*l1;
    }

    return obj;
}

/* instantiate for compilation */
template class Cp_d1_ql1b<double, uint32_t, uint16_t>;
template class Cp_d1_ql1b<float, uint32_t, uint16_t>;
template class Cp_d1_ql1b<double, uint32_t, uint32_t>;
template class Cp_d1_ql1b<float, uint32_t, uint32_t>;
