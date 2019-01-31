#include <cstdint>
#include "../include/cp_pfdr_d1_ql1b.hpp"

#include <ctime>

using namespace std;

/* index_t must be able to represent the numbers of vertices and of
 * (undirected) edges in the main graph; comp_t must be able to represent the
 * number of constant connected components in the reduced graph */
typedef uint32_t index_t;
typedef uint16_t comp_t;
/* uncomment the following if more than 65535 components are expected */
// typedef uint32_t comp_t;

/* template for handling both single and double precisions */
template<typename real_t>
static void cp_pfdr_d1_ql1b_test(index_t V, const real_t *Y, 
        index_t E, const index_t *first_edge, const index_t *adj_vertices,
        const real_t *edge_weights, real_t homo_edge_weight,
        comp_t* &Comp, int* &it, double* &Time, real_t* &Obj, real_t* &Dif, real_t* &rX)
{
    size_t N = DIAG_ATA;
    const real_t *A = nullptr; 
    const real_t a = 1.0;

    const real_t* Yl1 = nullptr;
    const real_t *l1_weights = nullptr;
    real_t homo_l1_weight = 0;
    const real_t *low_bnd = nullptr;
    real_t homo_low_bnd = -INF_REAL;
    const real_t *upp_bnd = nullptr;
    real_t homo_upp_bnd = INF_REAL;

    /* algorithmic parameters */
    real_t cp_dif_tol = 1e-4;
    int cp_it_max = 10;
    real_t pfdr_rho = 1.0;
    real_t pfdr_cond_min = 1e-2;
    real_t pfdr_dif_rcd = 0.0;
    real_t pfdr_dif_tol = 1e-3*cp_dif_tol;
    int pfdr_it_max = 1e4;
    int verbose = 1e3;

    /**  prepare output; rX is created later  **/

    if (Comp != nullptr) delete[] Comp;
    Comp = new comp_t[V];
    if (it != nullptr) delete it;
    it = new int;

    /**  cut-pursuit with preconditioned forward-Douglas-Rachford  **/

    Cp_d1_ql1b<real_t, index_t, comp_t> *cp =
        new Cp_d1_ql1b<real_t, index_t, comp_t>(V, E, first_edge, adj_vertices);

    cp->set_edge_weights(edge_weights, homo_edge_weight);
    cp->set_quadratic(Y, N, A, a);
    cp->set_l1(l1_weights, homo_l1_weight, Yl1);
    cp->set_bounds(low_bnd, homo_low_bnd, upp_bnd, homo_upp_bnd);
    cp->set_cp_param(cp_dif_tol, cp_it_max, verbose);
    cp->set_pfdr_param(pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_it_max,
        pfdr_dif_tol);
    cp->set_monitoring_arrays(Obj, Time, Dif);

    cp->set_components(0, Comp); // use the preallocated component array Comp

    *it = cp->cut_pursuit();

    /* copy reduced values */
    comp_t rV = cp->get_components();
    real_t *cp_rX = cp->get_reduced_values();
    if (rX != nullptr) delete[] rX;
    rX = new real_t[rV];
    for (comp_t rv = 0; rv < rV; rv++){ rX[rv] = cp_rX[rv]; }
    
    cp->set_components(0, nullptr); // prevent Comp to be free()'d
    delete cp;
}

double random_double(double min = 0.0, double max = 1.0){
    double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    return r*(max-min)+min;
}

void print_grid(double *array, int size){
    int ssize = (size<8)?size:8;
    for (int i = 0; i < ssize; ++i) {
        for (int j = 0; j < ssize; ++j) 
            std::cout << array[i*size+j] << '\t';
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]){
    srand(time(nullptr));
    /* input arguments */
    index_t size = 32;
    index_t V = size*size;
    double *Y = new double[V];
    index_t E = 4*size*(size-1);
    index_t *first_edge = new index_t[V+1];
    index_t *adj_vertices = new index_t[E];
    double *edge_weights = nullptr;
    double homo_edge_weight = 0.1;

    /* output arguments */
    comp_t* Comp = nullptr;
    int* it = nullptr;
    double* Time = nullptr;
    double* Obj = nullptr;
    double* Dif = nullptr;
    double* rX = nullptr;

    /* assigning random values to input Y */
    for (int i = 0; i < V; ++i)  Y[i] = random_double();

    /* build the grid/mesh graph */
    const int directions[4][2] = {{0,1},{0,-1},{1,0},{-1,0}};
    index_t edge_num = 0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            first_edge[i*size+j] = edge_num;
            for (int k = 0; k < 4; ++k) {
                if (i+directions[k][0] < size && i+directions[k][0] > -1 
                    && j+directions[k][1] < size && j+directions[k][1] > -1) {
                    adj_vertices[edge_num] = (i+directions[k][0])*32 + j+directions[k][1];
                    ++edge_num;
                }
            }
        }
    }
    first_edge[V] = edge_num;

    print_grid(Y,size);
    
    cp_pfdr_d1_ql1b_test<double>(V,Y,E,first_edge,adj_vertices,edge_weights,homo_edge_weight,Comp,it,Time,Obj,Dif,rX);

    double *X = new double[V];
    for (int i = 0;i < V; ++i) X[i] = rX[Comp[i]];

    print_grid(X,size);

    delete[] Y;
    delete[] first_edge;
    delete[] adj_vertices;
    delete[] Comp;
    delete it;
    delete[] rX;
    delete[] X;
}
