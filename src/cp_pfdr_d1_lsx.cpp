/*=============================================================================
 * Hugo Raguet 2018
 *===========================================================================*/
#include <cmath>
#include "../include/cp_pfdr_d1_lsx.hpp"
#include "../include/omp_num_threads.hpp"
#include "../include/matrix_tools.hpp"
#include "../include/pfdr_d1_lsx.hpp"

#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)
#define HALF ((real_t) 0.5)
#define INF_REAL (std::numeric_limits<real_t>::infinity())
#define EDGE_WEIGHTS_(e) (edge_weights ? edge_weights[(e)] : homo_edge_weight)
#define LOSS_WEIGHTS_(v) (loss_weights ? loss_weights[(v)] : ONE)
#define COOR_WEIGHTS_(d) (coor_weights ? coor_weights[(d)] : ONE)

#define TPL template <typename real_t, typename index_t, typename comp_t>
#define CP_D1_LSX Cp_d1_lsx<real_t, index_t, comp_t>

using namespace std;

TPL CP_D1_LSX::Cp_d1_lsx(index_t V, index_t E, const index_t* first_edge,
    const index_t* adj_vertices, size_t D, const real_t* Y) :
    Cp_d1<real_t, index_t, comp_t>(V, E, first_edge, adj_vertices, D, D11),
    Y(Y)
{
    if (numeric_limits<comp_t>::max() < D){
        cerr << "Cut-pursuit d1 loss simplex: comp_t must be able to represent"
            "the dimension D (" << D << ")." << endl;
        exit(EXIT_FAILURE);
    }

    loss = LINEAR;
    loss_weights = nullptr;

    pfdr_rho = 1.0; pfdr_cond_min = 1e-2; pfdr_dif_rcd = 0.0;
    pfdr_dif_tol = 1e-3*dif_tol; pfdr_it = pfdr_it_max = 1e4;

    /* with a separable loss, components are only coupled by total variation
     * and it makes sense to consider nonevolving components as saturated */
    monitor_evolution = true;
}

TPL void CP_D1_LSX::set_loss(real_t loss, const real_t* Y,
    const real_t* loss_weights)
{
    if (loss < ZERO || loss > ONE){
        cerr << "Cut-pursuit d1 loss simplex: loss parameter should be between"
            " 0 and 1 (" << loss << " given)." << endl;
        exit(EXIT_FAILURE);
    }
    this->loss = loss;
    if (Y){ this->Y = Y; }
    this->loss_weights = loss_weights; 
}

TPL void CP_D1_LSX::set_pfdr_param(real_t rho, real_t cond_min, real_t dif_rcd,
    int it_max, real_t dif_tol)
{
    this->pfdr_rho = rho;
    this->pfdr_cond_min = cond_min;
    this->pfdr_dif_rcd = dif_rcd;
    this->pfdr_it_max = it_max;
    this->pfdr_dif_tol = dif_tol;
}

TPL void CP_D1_LSX::solve_reduced_problem()
{
    if (rV == 1){ /**  single connected component  **/

        #pragma omp parallel for schedule(static) NUM_THREADS(D*V, D)
        for (size_t d = 0; d < D; d++){
            rX[d] = ZERO;
            size_t vd = d;
            for (index_t v = 0; v < V; v++){
                rX[d] += LOSS_WEIGHTS_(v)*Y[vd];
                vd += D;
            }
        }

        if (loss == LINEAR){ /* optimum at simplex corner */
            size_t idx = 0;
            real_t max = rX[idx];
            for (size_t d = 1; d < D; d++){
                if (rX[d] > max){ max = rX[idx = d]; }
            }
            for (size_t d = 0; d < D; d++){ rX[d] = d == idx ? ONE : ZERO; }
        }else{ /* optimum at barycenter */
            real_t total_weight = ZERO;
            #pragma omp parallel for schedule(static) NUM_THREADS(V) \
                reduction(+:total_weight)
            for (index_t v = 0; v < V; v++){
                total_weight += LOSS_WEIGHTS_(v);
            }
            for (size_t d = 0; d < D; d++){ rX[d] /= total_weight; }
        }

    }else{ /**  preconditioned forward-Douglas-Rachford  **/

        /* compute reduced observation and weights */
        real_t* rY = (real_t*) malloc_check(sizeof(real_t)*D*rV);
        real_t* reduced_loss_weights =
            (real_t*) malloc_check(sizeof(real_t)*rV);
        #pragma omp parallel for schedule(dynamic) NUM_THREADS(V, rV)
        for (comp_t rv = 0; rv < rV; rv++){
            real_t *rYv = rY + rv*D;
            for (size_t d = 0; d < D; d++){ rYv[d] = ZERO; }
            reduced_loss_weights[rv] = ZERO;
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                index_t v = comp_list[i];
                const real_t *Yv = Y + v*D;
                for (size_t d = 0; d < D; d++){
                    rYv[d] += LOSS_WEIGHTS_(v)*Yv[d];
                }
                reduced_loss_weights[rv] += LOSS_WEIGHTS_(v);
            }
            for (size_t d = 0; d < D; d++){
                rYv[d] /= reduced_loss_weights[rv];
            }
        }

        Pfdr_d1_lsx<real_t, comp_t> *pfdr = new Pfdr_d1_lsx<real_t, comp_t>(
                rV, rE, reduced_edges, loss, D, rY, coor_weights);

        pfdr->set_edge_weights(reduced_edge_weights);
        pfdr->set_loss(reduced_loss_weights);
        pfdr->set_conditioning_param(pfdr_cond_min, pfdr_dif_rcd);
        pfdr->set_relaxation(pfdr_rho);
        pfdr->set_algo_param(pfdr_dif_tol, pfdr_it_max, verbose);
        pfdr->set_iterate(rX);
        pfdr->initialize_iterate();

        pfdr_it = pfdr->precond_proximal_splitting();

        pfdr->set_iterate(nullptr); // prevent rX to be free()'d at deletion
        delete pfdr;

        free(rY); free(reduced_loss_weights);
    }
}

TPL uintmax_t CP_D1_LSX::split_complexity()
{
    uintmax_t complexity = maxflow_complexity(); // graph cut
    complexity += V; // account for gradient and labeling
    complexity += 2*E; // edges capacities
    complexity *= D - 1; // D - 1 alternative ascent coordinates
    return complexity*(V - saturated_vert)/V; // account saturation linearly
}

TPL index_t CP_D1_LSX::split()
{
    grad = (real_t*) malloc_check(sizeof(real_t)*D*V);

    const real_t c = (ONE - loss), q = loss/D, r = q/c; // useful for KLs

    uintmax_t Vns = V - saturated_vert;
    uintmax_t num_ops = D*Vns*(loss == LINEAR || loss == QUADRATIC ? 1 : 3);
    num_ops += E*Vns/V;
    num_ops += Vns/V;

    #pragma omp parallel for schedule(static) NUM_THREADS(num_ops, V)
    for (index_t v = 0; v < V; v++){
        comp_t rv = comp_assign[v];
        if (saturation(rv)){ continue; }

        real_t* gradv = grad + D*v;
        real_t* rXv = rX + D*rv;

        /**  gradient of differentiable loss term  **/
        const real_t* Yv = Y + D*v;
        for (size_t d = 0; d < D; d++){
            if (loss == LINEAR){ /* linear loss, grad = - w Y */
                gradv[d] = -LOSS_WEIGHTS_(v)*Yv[d];
            }else if (loss == QUADRATIC){ /* quadratic loss, grad = w(X - Y) */
                gradv[d] = LOSS_WEIGHTS_(v)*(rXv[d] - Yv[d]);
            }else{ /* dKLs/dx_k = -(1-s)(s/D + (1-s)y_k)/(s/D + (1-s)x_k) */
                gradv[d] = -LOSS_WEIGHTS_(v)*(q + c*Yv[d])/(r + rXv[d]);
            }
        }

        /**  differentiable d1 contribution  **/ 
        for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
            if (is_active(e)){
                index_t u = adj_vertices[e];
                real_t* rXu = rX + comp_assign[u]*D;
                real_t* gradu = grad + u*D;
                for (size_t d = 0; d < D; d++){
                    real_t grad_d1 = (rXv[d] - rXu[d] > eps ?
                        EDGE_WEIGHTS_(e) : -EDGE_WEIGHTS_(e))*COOR_WEIGHTS_(d);
                    gradv[d] += grad_d1;
                    gradu[d] -= grad_d1;
                    /* equality of _some_ coordinates constitutes a source of
                     * nondifferentiability; this is actually not taken into
                     * account, see below */
                }
            }
        }
    }

    index_t activation = Cp<real_t, index_t, comp_t>::split();

    free(grad);
    return activation;
}

TPL void CP_D1_LSX::split_component(Cp_graph<real_t, index_t, comp_t>* G,
    comp_t rv)
{
    /**  directions are searched in the set \prod_v Dv, where for each vertex,
     * Dv = {1d - 1dmv in R^D | d in {1,...,D}}, with dmv in argmax_d' {x_vd'}
     * that is to say dmv is a coordinate with maximum value, and it is tested
     * against all alternative coordinates; an approximate solution is
     * searched with one alpha-expansion cycle  **/

    /* find coordinate with maximum value */
    comp_t dmv = 0;
    real_t* rXv = rX + rv*D;
    real_t max = rXv[0];
    for (comp_t d = 1; d < D; d++){ if (rXv[d] > max){ max = rXv[dmv = d]; } }

    /* initialize best ascent coordinate at the coordinate with maximum
     * value, corresponding to a null descent direction (1dmv - 1dmv) */
    for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
        label_assign[comp_list[i]] = dmv;
    }

    /* iterate over all D - 1 alternative ascent coordinates */
    for (comp_t d_alt = 1; d_alt < D; d_alt++){
        /* actual ascent direction */
        comp_t d = d_alt == dmv ? 0 : d_alt;

        /* set the source/sink capacities */
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            real_t* gradv = grad + v*D;
            /* unary cost for changing current dir_v to 1d - 1dmv */
            term_capacities(v) = gradv[d] - gradv[label_assign[v]];
        }

        /* set d1 edge capacities within each component;
         * strictly speaking, active edges should not be directly ignored,
         * because _some_ neighboring coordinates can still be equal, yielding
         * nondifferentiability and thus corresponding to positive capacities;
         * however, such capacities are somewhat cumbersome to compute, and 
         * more importantly max flows cannot be easily computed in parallel,
         * since the components would not be independent anymore;
         * we thus stick with the current heuristic for now */
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t u = comp_list[i];
            for (index_t e = first_edge[u]; e < first_edge[u + 1]; e++){
                if (!is_free(e)){ continue; }
                index_t v = adj_vertices[e];
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
                /* current ascent coordinate */
                comp_t du = label_assign[u];
                comp_t dv = label_assign[v];
                /* A = E(0,0) is the cost of the current ascent coords */
                real_t A = du == dv ? ZERO : EDGE_WEIGHTS_(e)
                    *(COOR_WEIGHTS_(du) + COOR_WEIGHTS_(dv));
                /* B = E(0,1) is the cost of changing dv to d */
                real_t B = du == d ? ZERO : EDGE_WEIGHTS_(e)
                    *(COOR_WEIGHTS_(du) + COOR_WEIGHTS_(d));
                /* C = E(1,0) is the cost of changing du to d */
                real_t C = dv == d ? ZERO : EDGE_WEIGHTS_(e)
                    *(COOR_WEIGHTS_(dv) + COOR_WEIGHTS_(d));
                /* D = E(1,1) = 0 is for changing both du and dv to d */
                /* set weights in accordance with orientation u -> v */
                term_capacities(u) += C - A;
                term_capacities(v) -= C;
                set_edge_capacities(e, B + C - A, ZERO);
            }
        }

        /* find min cut and update best ascent coordinates accordingly */
        G->maxflow(first_vertex[rv + 1] - first_vertex[rv], comp_list +
            first_vertex[rv]);
        
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            if (is_sink(v)){ label_assign[v] = d; }
        }
    } // end for d_alt
}

TPL real_t CP_D1_LSX::compute_evolution(bool compute_dif)
{
    index_t num_ops = compute_dif ? D*(V - saturated_vert) : D*saturated_comp;
    real_t dif = ZERO;
    /* auxiliary variable for parallel region */
    comp_t saturated_comp_par = 0; 
    index_t saturated_vert_par = 0;
    #pragma omp parallel for schedule(dynamic) NUM_THREADS(num_ops, rV) \
        reduction(+:dif, saturated_comp_par, saturated_vert_par)
    for (comp_t rv = 0; rv < rV; rv++){
        real_t* rXv = rX + rv*D;
        if (saturation(rv)){
            real_t* lrXv = last_rX +
                tmp_comp_assign(comp_list[first_vertex[rv]])*D;
            real_t rv_dif = ZERO;
            for (size_t d = 0; d < D; d++){ rv_dif += abs(lrXv[d] - rXv[d]); }
            if (rv_dif > dif_tol){
                saturation(rv) = false;
            }else{
                saturated_comp_par++;
                saturated_vert_par += first_vertex[rv + 1] - first_vertex[rv];
            }
            if (compute_dif){
                dif += rv_dif*(first_vertex[rv + 1] - first_vertex[rv]);
            }
        }else if (compute_dif){
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                real_t* lrXv = last_rX + tmp_comp_assign(comp_list[i])*D;
                for (size_t d = 0; d < D; d++){ dif += abs(lrXv[d] - rXv[d]); }
            }
        }
    }
    saturated_comp = saturated_comp_par;
    saturated_vert = saturated_vert_par;
    return compute_dif ? dif/V : INF_REAL;
}

TPL real_t CP_D1_LSX::compute_objective()
/* unfortunately, at this point one does not have access to the reduced objects
 * computed in the routine solve_reduced_problem() */
{
    real_t obj = ZERO;

    if (loss == LINEAR){
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
            reduction(+:obj)
        for (index_t v = 0; v < V; v++){
            real_t* rXv = rX + comp_assign[v]*D;
            const real_t* Yv = Y + v*D;
            real_t prod = ZERO;
            for (size_t d = 0; d < D; d++){ prod += rXv[d]*Yv[d]; }
            obj -= LOSS_WEIGHTS_(v)*prod;
        }
    }else if (loss == QUADRATIC){
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
            reduction(+:obj)
        for (index_t v = 0; v < V; v++){
            real_t* rXv = rX + comp_assign[v]*D;
            const real_t* Yv = Y + v*D;
            real_t dif2 = ZERO;
            for (size_t d = 0; d < D; d++){
                dif2 += (rXv[d] - Yv[d])*(rXv[d] - Yv[d]);
            }
            obj += LOSS_WEIGHTS_(v)*dif2;
        }
        obj *= HALF;
    }else{ /* smoothed Kullback-Leibler */
        const real_t c = (ONE - loss);
        const real_t q = loss/D;
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
            reduction(+:obj) 
        for (index_t v = 0; v < V; v++){
            real_t* rXv = rX + comp_assign[v]*D;
            const real_t* Yv = Y + v*D;
            real_t KLs = ZERO;
            for (size_t d = 0; d < D; d++){
                real_t ys = q + c*Yv[d];
                KLs += ys*log(ys/(q + c*rXv[d]));
            }
            obj += LOSS_WEIGHTS_(v)*KLs;
        }
    }

    obj += compute_graph_d1(); // ||x||_d1

    return obj;
}

/* instantiate for compilation */
template class Cp_d1_lsx<float, uint32_t, uint16_t>;
template class Cp_d1_lsx<double, uint32_t, uint16_t>;
template class Cp_d1_lsx<float, uint32_t, uint32_t>;
template class Cp_d1_lsx<double, uint32_t, uint32_t>;
