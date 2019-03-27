/*=============================================================================
 * [Comp, rX, it, Obj, Time, Dif] = cp_pfdr_d1_lsx_mex(loss, Y, first_edge,
 *      adj_vertices, edge_weights = 1.0, loss_weights = [],
 *      d1_coor_weights = [], cp_dif_tol = 1e-3, cp_it_max = 10,
 *      pfdr_rho = 1.0, pfdr_cond_min = 1e-2, pfdr_dif_rcd = 0.0,
 *      pfdr_dif_tol = 1e-3*cp_dif_tol, pfdr_it_max = 1e4, verbose = 1e2)
 * 
 *  Hugo Raguet 2016, 2018
 *===========================================================================*/
#include <cstdint>
#include "mex.h"
#include "../../include/cp_pfdr_d1_lsx.hpp"

using namespace std;

/* index_t must be able to represent the number of vertices and of (undirected)
 * edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph, as well as the dimension D */
typedef uint32_t index_t;
# define VERTEX_CLASS mxUINT32_CLASS
# define VERTEX_ID "uint32"
typedef uint16_t comp_t;
# define COMP_CLASS mxUINT16_CLASS
# define COMP_ID "uint16"
/* uncomment the following if more than 65535 components are expected */
// typedef uint32_t comp_t;
// # define COMP_CLASS mxUINT32_CLASS
// # define COMP_ID "uint32"

/* arrays with arguments type */
static const int args_real_t[] = {1, 4, 5, 6};
static const int n_real_t = 4;
static const int args_index_t[] = {2, 3};
static const int n_index_t = 2;

/* function for checking arguments type */
static void check_args(int nrhs, const mxArray *prhs[], const int* args,
    int n, mxClassID id, const char* id_name)
{
    for (int i = 0; i < n; i++){
        if (nrhs > args[i] && mxGetClassID(prhs[args[i]]) != id
            && mxGetNumberOfElements(prhs[args[i]]) > 1){
            mexErrMsgIdAndTxt("MEX", "Cut-pursuit d1 loss simplex: "
                "argument %d is of class %s, but class %s is expected.",
                args[i] + 1, mxGetClassName(prhs[args[i]]), id_name);
        }
    }
}

/* resize memory buffer allocated by mxMalloc and create a row vector */
template <typename type_t>
static mxArray* resize_and_create_mxRow(type_t* buffer, size_t size,
    mxClassID id)
{
    mxArray* row = mxCreateNumericMatrix(0, 0, id, mxREAL);
    if (size){
        mxSetM(row, 1);
        mxSetN(row, size);
        buffer = (type_t*) mxRealloc((void*) buffer, sizeof(type_t)*size);
        mxSetData(row, (void*) buffer);
    }else{
        mxFree((void*) buffer);
    }
    return row;
}

/* template for handling both single and double precisions */
template<typename real_t>
static void cp_pfdr_d1_lsx_mex(int nlhs, mxArray **plhs, int nrhs, \
    const mxArray **prhs)
{
    /**  get inputs  **/

    /* sizes and loss */
    real_t loss = mxGetScalar(prhs[0]);
    size_t D = mxGetM(prhs[1]);
    index_t V = mxGetN(prhs[1]);
    const real_t *Y = (real_t*) mxGetData(prhs[1]);
    const real_t *loss_weights = (nrhs > 5 && !mxIsEmpty(prhs[5])) ?
        (real_t*) mxGetData(prhs[5]) : nullptr;

    /* graph structure */
    index_t E = mxGetNumberOfElements(prhs[3]);
    check_args(nrhs, prhs, args_index_t, n_index_t, VERTEX_CLASS, VERTEX_ID);
    const index_t *first_edge = (index_t*) mxGetData(prhs[2]);
    const index_t *adj_vertices = (index_t*) mxGetData(prhs[3]);
    if (mxGetNumberOfElements(prhs[2]) != (V + 1)){
        mexErrMsgIdAndTxt("MEX", "Cut-pursuit d1 loss simplex: "
            "argument 3 'adj_vertices' should contain |V|+1 = %d elements, "
            "but %d are given.", (V + 1), mxGetNumberOfElements(prhs[2]));
    }

    /* penalizations */
    const real_t *edge_weights =
        (nrhs > 4 && mxGetNumberOfElements(prhs[4]) > 1) ?
        (real_t*) mxGetData(prhs[4]) : nullptr;
    real_t homo_edge_weight =
        (nrhs > 4 && mxGetNumberOfElements(prhs[4]) == 1) ?
        mxGetScalar(prhs[4]) : 1;
    const real_t *d1_coor_weights = nrhs > 6 && !mxIsEmpty(prhs[6]) ?
        (real_t*) mxGetData(prhs[6]) : nullptr;

    real_t cp_dif_tol = (nrhs > 7) ? mxGetScalar(prhs[7]) : 1e-3;
    int cp_it_max = (nrhs > 8) ? mxGetScalar(prhs[8]) : 10;
    real_t pfdr_rho = (nrhs > 9) ? mxGetScalar(prhs[9]) : 1.0;
    real_t pfdr_cond_min = (nrhs > 10) ? mxGetScalar(prhs[10]) : 1e-2;
    real_t pfdr_dif_rcd = (nrhs > 11) ? mxGetScalar(prhs[11]) : 0.0;
    real_t pfdr_dif_tol = (nrhs > 12) ? mxGetScalar(prhs[12]) : 1e-3*cp_dif_tol;
    int pfdr_it_max = (nrhs > 13) ? mxGetScalar(prhs[13]) : 1e4;
    int verbose = (nrhs > 14) ? mxGetScalar(prhs[14]) : 1e3;

    /**  prepare output; rX (plhs[1]) is created later  **/

    plhs[0] = mxCreateNumericMatrix(1, V, COMP_CLASS, mxREAL);
    comp_t *Comp = (comp_t*) mxGetData(plhs[0]);
    plhs[2] = mxCreateNumericMatrix(1, 1, VERTEX_CLASS, mxREAL);
    int *it = (int*) mxGetData(plhs[2]);


    real_t* Obj = nlhs > 3 ?
        (real_t*) mxMalloc(sizeof(real_t)*(cp_it_max + 1)) : nullptr;
    double* Time = nlhs > 4 ?
        (double*) mxMalloc(sizeof(double)*(cp_it_max + 1)) : nullptr;
    real_t *Dif = nlhs > 5 ?
        (real_t*) mxMalloc(sizeof(double)*cp_it_max) : nullptr;

    /**  cut-pursuit with preconditioned forward-Douglas-Rachford  **/

    Cp_d1_lsx<real_t, index_t, comp_t> *cp =
        new Cp_d1_lsx<real_t, index_t, comp_t>
            (V, E, first_edge, adj_vertices, D, Y);

    cp->set_loss(loss, Y, loss_weights);
    cp->set_edge_weights(edge_weights, homo_edge_weight, d1_coor_weights);
    cp->set_monitoring_arrays(Obj, Time, Dif);
    cp->set_components(0, Comp);
    cp->set_cp_param(cp_dif_tol, cp_it_max, verbose);
    cp->set_pfdr_param(pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_it_max,
        pfdr_dif_tol);

    *it = cp->cut_pursuit();

    /* copy reduced values */
    comp_t rV = cp->get_components();
    real_t *cp_rX = cp->get_reduced_values();
    plhs[1] = mxCreateNumericMatrix(D, rV, mxGetClassID(prhs[1]), mxREAL);
    real_t *rX = (real_t*) mxGetData(plhs[1]);
    for (size_t rvd = 0; rvd < rV*D; rvd++){ rX[rvd] = cp_rX[rvd]; }
    
    cp->set_components(0, nullptr); // prevent Comp to be free()'d
    delete cp;

    /**  resize monitoring arrays and assign to outputs  **/
    if (nlhs > 3){
        plhs[3] = resize_and_create_mxRow(Obj, *it + 1, mxGetClassID(prhs[1]));
    }
    if (nlhs > 4){
        plhs[4] = resize_and_create_mxRow(Time, *it + 1, mxDOUBLE_CLASS);
    }
    if (nlhs > 5){
        plhs[5] = resize_and_create_mxRow(Dif, *it, mxGetClassID(prhs[1]));
    }

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    /* real type is determined by second parameter Y */
    if (mxIsDouble(prhs[1])){
        check_args(nrhs, prhs, args_real_t, n_real_t, mxDOUBLE_CLASS,
            "double");
        cp_pfdr_d1_lsx_mex<double>(nlhs, plhs, nrhs, prhs);
    }else{
        check_args(nrhs, prhs, args_real_t, n_real_t, mxSINGLE_CLASS,
            "single");
        cp_pfdr_d1_lsx_mex<float>(nlhs, plhs, nrhs, prhs);
    }
}
