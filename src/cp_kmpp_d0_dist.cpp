/*=============================================================================
 * Hugo Raguet 2018
 *===========================================================================*/
#include <random>
#include "../include/cp_kmpp_d0_dist.hpp"
#include "../include/omp_num_threads.hpp"

#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)
#define HALF ((real_t) 0.5)
#define INF_REAL (std::numeric_limits<real_t>::infinity())
#define VERT_WEIGHTS_(v) (vert_weights ? vert_weights[(v)] : ONE)

#define TPL template <typename real_t, typename index_t, typename comp_t>
#define CP_D0_DIST Cp_d0_dist<real_t, index_t, comp_t>

using namespace std;

TPL CP_D0_DIST::Cp_d0_dist(index_t V, index_t E, const index_t* first_edge,
    const index_t* adj_vertices, const real_t* Y, size_t D) :
    Cp_d0<real_t, index_t, comp_t>(V, E, first_edge, adj_vertices, D), Y(Y)
{
    vert_weights = coor_weights = nullptr;
    comp_weights = nullptr; 
    kmpp_init_num = 3;
    kmpp_iter_num = 3;

    loss = QUADRATIC;
    fYY = ZERO;
    fXY = INF_REAL;
}

TPL CP_D0_DIST::~Cp_d0_dist(){ free(comp_weights); }

TPL void CP_D0_DIST::set_loss(real_t loss, const real_t* Y,
    const real_t* vert_weights, const real_t* coor_weights)
{
    if (loss < ZERO || loss > ONE){
        cerr << "Cut-pursuit d0 distance: loss parameter should be between"
            " 0 and 1 (" << loss << " given)." << endl;
        exit(EXIT_FAILURE);
    }
    if (loss == ZERO){ loss = eps; } // avoid singularities
    this->loss = loss;
    if (Y){ this->Y = Y; }
    this->vert_weights = vert_weights;
    this->coor_weights = coor_weights; 
    /* recompute the constant dist(Y, Y) if necessary */
    fYY = ZERO;
    if (loss != QUADRATIC){
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
            reduction(+:fYY)
        for (index_t v = 0; v < V; v++){
            const real_t* Yv = Y + D*v;
            fYY += VERT_WEIGHTS_(v)*distance(Yv, Yv);
        }
    }
}

TPL void CP_D0_DIST::set_kmpp_param(int kmpp_init_num, int kmpp_iter_num)
{
    this->kmpp_init_num = kmpp_init_num;
    this->kmpp_iter_num = kmpp_iter_num;
}

TPL real_t CP_D0_DIST::fv(index_t v, const real_t* Xv)
{ return VERT_WEIGHTS_(v)*distance(Y + D*v, Xv); }

TPL real_t CP_D0_DIST::compute_f()
{
    if (fXY == INF_REAL){ fXY = Cp_d0<real_t, index_t, comp_t>::compute_f(); }
    return fXY - fYY;
}

TPL void CP_D0_DIST::solve_reduced_problem()
{
    free(comp_weights);
    comp_weights = (real_t*) malloc_check(sizeof(real_t)*rV);
    fXY = INF_REAL; // rX will change, fXY must be recomputed

    #pragma omp parallel for schedule(static) NUM_THREADS(2*D*V, rV)
    for (comp_t rv = 0; rv < rV; rv++){
        real_t* rXv = rX + D*rv;
        comp_weights[rv] = ZERO;
        for (size_t d = 0; d < D; d++){ rXv[d] = ZERO; }
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            comp_weights[rv] += VERT_WEIGHTS_(v);
            const real_t* Yv = Y + D*v;
            for (size_t d = 0; d < D; d++){ rXv[d] += VERT_WEIGHTS_(v)*Yv[d]; }
        }
        if (comp_weights[rv]){
            for (size_t d = 0; d < D; d++){ rXv[d] /= comp_weights[rv]; }
        } /* maybe one should raise an exception for zero weight component */
    }
}

TPL void CP_D0_DIST::init_split_values(comp_t rv, real_t* altX,
    comp_t* label_assign)
{
    index_t comp_size = first_vertex[rv + 1] - first_vertex[rv];

    /* distance map and random device for k-means++ */
    real_t* nearest_dist = (real_t*) malloc_check(sizeof(real_t)*comp_size);
    default_random_engine rand_gen; // default seed also enough for our purpose

    /* current centroids and best sum of distances */
    real_t* centroids = (real_t*) malloc_check(sizeof(real_t)*D*K);
    real_t min_sum_dist = INF_REAL;

    /* store centroids entropy for Kullback-Leibler divergence */
    real_t* bottom_dist = loss == QUADRATIC ?
        nullptr : (real_t*) malloc_check(sizeof(real_t)*K);

    /**  kmeans ++  **/
    for (int kmpp_init = 0; kmpp_init < kmpp_init_num; kmpp_init++){

        /**  initialization  **/ 
        for (comp_t k = 0; k < K; k++){
            index_t rand_i;
            if (k == 0){
                uniform_int_distribution<index_t> unif_distr(0, comp_size - 1);
                rand_i = unif_distr(rand_gen);
            }else{
                for (index_t i = 0; i < comp_size; i++){
                    index_t v = comp_list[first_vertex[rv] + i];
                    nearest_dist[i] = INF_REAL;
                    for (comp_t l = 0; l < k; l++){
                        real_t dist = distance(centroids + D*l, Y + D*v);
                        if (loss != QUADRATIC){ dist -= bottom_dist[l]; }
                        if (dist < nearest_dist[i]){ nearest_dist[i] = dist; }
                    }
                    if (vert_weights){ nearest_dist[i] *= vert_weights[v]; }
                }
                discrete_distribution<index_t> dist_distr(nearest_dist,
                    nearest_dist + comp_size);
                rand_i = dist_distr(rand_gen);
            }
            index_t rand_v = comp_list[first_vertex[rv] + rand_i];
            const real_t* Yv = Y + D*rand_v;
            real_t* Ck = centroids + D*k;
            for (size_t d = 0; d < D; d++){ Ck[d] = Yv[d]; }
            if (loss != QUADRATIC){ bottom_dist[k] = distance(Ck, Ck); }
        } // end for k

        /**  k-means  **/
        for (int kmpp_iter = 0; kmpp_iter < kmpp_iter_num; kmpp_iter++){
            /* assign clusters to centroids */
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                index_t v = comp_list[i];
                real_t min_dist = INF_REAL;
                for (comp_t k = 0; k < K; k++){
                    real_t dist = distance(centroids + D*k, Y + D*v);
                    if (dist < min_dist){
                        min_dist = dist;
                        label_assign[v] = k;
                    }
                }
            }
            /* update centroids of clusters */
            update_split_values(rv, centroids, label_assign);
        }

        /**  compare resulting sum of distances and keep the best one  **/
        real_t sum_dist = ZERO;
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            comp_t k = label_assign[v];
            sum_dist += VERT_WEIGHTS_(v)*distance(centroids + D*k, Y + D*v);
        }
        if (sum_dist < min_sum_dist){
            min_sum_dist = sum_dist;
            for (size_t dk = 0; dk < D*K; dk++){ altX[dk] = centroids[dk]; }
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                index_t v = comp_list[i];
                set_tmp_comp_assign(v, label_assign[v]);
            }
        }

    } // end for kmpp_init

    free(bottom_dist);
    free(centroids);
    free(nearest_dist);

    /**  copy best label assignment  **/
    for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
        index_t v = comp_list[i];
        label_assign[v] = get_tmp_comp_assign(v);
    }
}

TPL void CP_D0_DIST::update_split_values(comp_t rv, real_t* altX,
    comp_t* label_assign)
{
    real_t* total_weights = (real_t*) malloc_check(sizeof(real_t)*K);
    for (comp_t k = 0; k < K; k++){
        total_weights[k] = ZERO;
        real_t* altXk = altX + D*k;
        for (size_t d = 0; d < D; d++){ altXk[d] = ZERO; }
    }
    for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
        index_t v = comp_list[i];
        comp_t k = label_assign[v];
        total_weights[k] += VERT_WEIGHTS_(v);
        const real_t* Yv = Y + D*v;
        real_t* altXk = altX + D*k;
        for (size_t d = 0; d < D; d++){ altXk[d] += VERT_WEIGHTS_(v)*Yv[d]; }
    }
    for (comp_t k = 0; k < K; k++){
        real_t* altXk = altX + D*k;
        if (total_weights[k]){
            for (size_t d = 0; d < D; d++){ altXk[d] /= total_weights[k]; }
        }else{ // no vertex assigned to k, flag with infinity
            altXk[0] = INF_REAL;
        }
    }
    free(total_weights);
}

TPL bool CP_D0_DIST::is_split_value(real_t altX){ return altX != INF_REAL; }

TPL void CP_D0_DIST::update_merge_candidate(size_t re, comp_t ru, comp_t rv)
{
    real_t* rXu = rX + D*ru;
    real_t* rXv = rX + D*rv;
    real_t wru = comp_weights[ru];
    real_t wrv = comp_weights[rv];

    if (loss == QUADRATIC){
        real_t gain = reduced_edge_weights[re]
            - wru*wrv/(wru + wrv)*distance(rXu, rXv);

        if (gain > ZERO){
            if (merge_info_list[re] == no_merge_info){
                merge_info_list[re] = new Merge_info(D);
            }
            merge_info_list[re]->gain = gain;
            wru /= (comp_weights[ru] + comp_weights[rv]);
            wrv /= (comp_weights[ru] + comp_weights[rv]);
            for (size_t d = 0; d < D; d++){
                merge_info_list[re]->value[d] = wru*rXu[d] + wrv*rXv[d];
            }
        }else if (merge_info_list[re] != no_merge_info){
            delete merge_info_list[re];
            merge_info_list[re] = no_merge_info;
        }
    }else{
        if (merge_info_list[re] == no_merge_info){
            merge_info_list[re] = new Merge_info(D);
        }
        real_t* value = merge_info_list[re]->value;

        wru /= (comp_weights[ru] + comp_weights[rv]);
        wrv /= (comp_weights[ru] + comp_weights[rv]);
        for (size_t d = 0; d < D; d++){ value[d] = wru*rXu[d] + wrv*rXv[d]; }

        /* in the following some computations might be saved by factoring
         * multiplications and logarithms, at the cost of readability */
        merge_info_list[re]->gain = reduced_edge_weights[re]
            + comp_weights[ru]*(distance(rXu, rXu) - distance(rXu, value))
            + comp_weights[rv]*(distance(rXv, rXv) - distance(rXv, value));

        if (merge_info_list[re]->gain <= ZERO){
            delete merge_info_list[re];
            merge_info_list[re] = no_merge_info;
        }
    }
}

TPL size_t CP_D0_DIST::update_merge_complexity()
{ return rE*2*D; /* each update is only linear in D */ }

TPL void CP_D0_DIST::accept_merge_candidate(size_t re, comp_t& ru, comp_t& rv)
{
    Cp_d0<real_t, index_t, comp_t>::accept_merge_candidate(re, ru, rv);
        // ru now the root of the merge chain
    comp_weights[ru] += comp_weights[rv];
}

TPL index_t CP_D0_DIST::merge()
{
    index_t deactivation = Cp_d0<real_t, index_t, comp_t>::merge();
    free(comp_weights); comp_weights = nullptr;
    return deactivation;
}

TPL real_t CP_D0_DIST::compute_evolution(bool compute_dif)
{
    if (!compute_dif){ return INF_REAL; }
    real_t dif = ZERO;
    #pragma omp parallel for schedule(dynamic) reduction(+:dif) \
        NUM_THREADS(D*V*(rV - saturation_count)/rV, rV)
    for (comp_t rv = 0; rv < rV; rv++){
        if (is_saturated(rv)){ continue; }
        real_t* rXv = rX + D*rv;
        real_t distXX = loss == QUADRATIC ? ZERO : distance(rXv, rXv);
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            real_t* lrXv = last_rX + D*get_tmp_comp_assign(v);
            dif += VERT_WEIGHTS_(v)*(distance(rXv, lrXv) - distXX);
        }
    }
    real_t amp = compute_f();
    return amp > eps ? dif/amp : dif/eps;
}

/* instantiate for compilation */
template class Cp_d0_dist<float, uint32_t, uint16_t>;
template class Cp_d0_dist<double, uint32_t, uint16_t>;
template class Cp_d0_dist<float, uint32_t, uint32_t>;
template class Cp_d0_dist<double, uint32_t, uint32_t>;
