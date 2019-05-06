/*=============================================================================
 * [Comp, rX, it, Obj, Time, Dif] = cp_pfdr_d1_ql1b_mex(Y | AtY, A | AtA,
 *      first_edge, adj_vertices, edge_weights = 1.0, Yl1 = [],
 *      l1_weights = 0.0, low_bnd = -Inf, upp_bnd = Inf, cp_dif_tol = 1e-4,
 *      cp_it_max = 10, pfdr_rho = 1.0, pfdr_cond_min = 1e-2,
 *      pfdr_dif_rcd = 0.0, pfdr_dif_tol = 1e-3*cp_dif_tol, pfdr_it_max = 1e4,
 *      verbose = 1e3, max_num_threads = 0, balance_parallel_split = true,
 *      AtA_if_square = true)
 * 
 *  Hugo Raguet 2016, 2018, 2019
 *===========================================================================*/
#include <cstdint>
#include "mex.h"
#include "../../include/cp_pfdr_d1_ql1b.hpp"

using namespace std;

/* index_t must be able to represent the number of vertices and of (undirected)
 * edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph */
typedef uint32_t index_t;
# define VERTEX_CLASS mxUINT32_CLASS
# define VERTEX_ID "uint32"
// typedef uint16_t comp_t;
// # define COMP_CLASS mxUINT16_CLASS
// # define COMP_ID "uint16"
/* uncomment the following if more than 65535 components are expected */
typedef uint32_t comp_t;
# define COMP_CLASS mxUINT32_CLASS
# define COMP_ID "uint32"

/* arrays with arguments type */
static const int args_real_t[] = {0, 1, 4, 5, 6, 7, 8};
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
            mexErrMsgIdAndTxt("MEX", "Cut-pursuit d1 quadratic l1 bounds: "
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
template <typename real_t, mxClassID mxREAL_CLASS>
static void cp_pfdr_d1_ql1b_mex(int nlhs, mxArray **plhs, int nrhs, \
    const mxArray **prhs)
{
    /**  get inputs  **/

    /* quadratic functional */
    size_t N = mxGetM(prhs[1]);
    index_t V = mxGetN(prhs[1]);

    if (N == 0 && V == 0){
        mexErrMsgIdAndTxt("MEX", "Cut-pursuit d1 quadratic l1 bounds: "
            "argument A cannot be empty.");
    }

    const real_t* Y = !mxIsEmpty(prhs[0]) ?
        (real_t*) mxGetData(prhs[0]) : nullptr;
    const real_t* A = (N == 1 && V == 1) ?
        nullptr : (real_t*) mxGetData(prhs[1]);
    const real_t a = (N == 1 && V == 1) ?  mxGetScalar(prhs[1]) : 1.0;

    if (V == 1){ /* quadratic functional is only weighted square difference */
        if (N == 1){
            if (!mxIsEmpty(prhs[0])){ /* fidelity is square l2 */
                V = mxGetNumberOfElements(prhs[0]);
            }else if (!mxIsEmpty(prhs[5])){ /* fidelity is only l1 */
                V = mxGetNumberOfElements(prhs[5]);
            }else{ /* should not happen */
                mexErrMsgIdAndTxt("MEX", "Cut-pursuit d1 quadratic l1 bounds: "
                    "arguments Y and Yl1 cannot be both empty.");
            }
        }else{ /* A is given V-by-1, representing a diagonal V-by-V */
            V = N;
        }
        N = DIAG_ATA;
    }else if (V == N && (nrhs < 20 || mxIsLogicalScalarTrue(prhs[19]))){
        N = FULL_ATA; // A and Y are left-premultiplied by A^t
    }

    /* graph structure */
    index_t E = mxGetNumberOfElements(prhs[3]);
    check_args(nrhs, prhs, args_index_t, n_index_t, VERTEX_CLASS, VERTEX_ID);
    const index_t *first_edge = (index_t*) mxGetData(prhs[2]);
    const index_t *adj_vertices = (index_t*) mxGetData(prhs[3]);
    if (mxGetNumberOfElements(prhs[2]) != (V + 1)){
        mexErrMsgIdAndTxt("MEX", "Cut-pursuit d1 quadratic l1 bounds: "
            "argument 3 'first_edge' should contain |V| + 1 = %d elements, "
            "but %d are given.", (V + 1), mxGetNumberOfElements(prhs[2]));
    }

    /* penalizations */
    const real_t* edge_weights =
        (nrhs > 4 && mxGetNumberOfElements(prhs[4]) > 1) ?
        (real_t*) mxGetData(prhs[4]) : nullptr;
    real_t homo_edge_weight =
        (nrhs > 4 && mxGetNumberOfElements(prhs[4]) == 1) ?
        mxGetScalar(prhs[4]) : 1.0;

    const real_t* Yl1 = (nrhs > 5 && !mxIsEmpty(prhs[5])) ?
        (real_t*) mxGetData(prhs[5]) : nullptr;
    const real_t* l1_weights =
        (nrhs > 6 && mxGetNumberOfElements(prhs[6]) > 1) ?
        (real_t*) mxGetData(prhs[6]) : nullptr;
    real_t homo_l1_weight =
        (nrhs > 6 && mxGetNumberOfElements(prhs[6]) == 1) ?
        mxGetScalar(prhs[6]) : 0.0;

    const real_t* low_bnd =
        (nrhs > 7 && mxGetNumberOfElements(prhs[7]) > 1) ?
        (real_t*) mxGetData(prhs[7]) : nullptr;
    real_t homo_low_bnd =
        (nrhs > 7 && mxGetNumberOfElements(prhs[7]) == 1) ?
        mxGetScalar(prhs[7]) : -INF_REAL;

    const real_t* upp_bnd =
        (nrhs > 8 && mxGetNumberOfElements(prhs[8]) > 1) ?
        (real_t*) mxGetData(prhs[8]) : nullptr;
    real_t homo_upp_bnd =
        (nrhs > 8 && mxGetNumberOfElements(prhs[8]) == 1) ?
        mxGetScalar(prhs[8]) : INF_REAL;

    /* algorithmic parameters */
    real_t cp_dif_tol = (nrhs > 9) ? mxGetScalar(prhs[9]) : 1e-4;
    int cp_it_max = (nrhs > 10) ? mxGetScalar(prhs[10]) : 10;
    real_t pfdr_rho = (nrhs > 11) ? mxGetScalar(prhs[11]) : 1.0;
    real_t pfdr_cond_min = (nrhs > 12) ? mxGetScalar(prhs[12]) : 1e-2;
    real_t pfdr_dif_rcd = (nrhs > 13) ? mxGetScalar(prhs[13]) : 0.0;
    real_t pfdr_dif_tol = (nrhs > 14) ?
        mxGetScalar(prhs[14]) : 1e-3*cp_dif_tol;
    int pfdr_it_max = (nrhs > 15) ? mxGetScalar(prhs[15]) : 1e4;
    int verbose = (nrhs > 16) ? mxGetScalar(prhs[16]) : 1e3;
    int max_num_threads = (nrhs > 17 && mxGetScalar(prhs[17]) > 0) ?
        mxGetScalar(prhs[17]) : omp_get_max_threads();
    bool balance_parallel_split = (nrhs > 18) ?
        mxIsLogicalScalarTrue(prhs[18]) : true;

    /**  prepare output; rX (plhs[1]) is created later  **/

    plhs[0] = mxCreateNumericMatrix(1, V, COMP_CLASS, mxREAL);
    comp_t *Comp = (comp_t*) mxGetData(plhs[0]);
    plhs[2] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    int *it = (int*) mxGetData(plhs[2]);

    real_t* Obj = nlhs > 3 ?
        (real_t*) mxMalloc(sizeof(real_t)*(cp_it_max + 1)) : nullptr;
    double* Time = nlhs > 4 ?
        (double*) mxMalloc(sizeof(double)*(cp_it_max + 1)) : nullptr;
    real_t *Dif = nlhs > 5 ?
        (real_t*) mxMalloc(sizeof(double)*cp_it_max) : nullptr;

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
    cp->set_parallel_param(max_num_threads, balance_parallel_split);
    cp->set_monitoring_arrays(Obj, Time, Dif);

    cp->set_components(0, Comp); // use the preallocated component array Comp

    *it = cp->cut_pursuit();

    /* copy reduced values */
    comp_t rV = cp->get_components();
    real_t* cp_rX = cp->get_reduced_values();
    plhs[1] = mxCreateNumericMatrix(rV, 1, mxREAL_CLASS, mxREAL);
    real_t* rX = (real_t*) mxGetData(plhs[1]);
    for (comp_t rv = 0; rv < rV; rv++){ rX[rv] = cp_rX[rv]; }
    
    cp->set_components(0, nullptr); // prevent Comp to be free()'d
    delete cp;

    /**  resize monitoring arrays and assign to outputs  **/
    if (nlhs > 3){
        plhs[3] = resize_and_create_mxRow(Obj, *it + 1, mxREAL_CLASS);
    }
    if (nlhs > 4){
        plhs[4] = resize_and_create_mxRow(Time, *it + 1, mxDOUBLE_CLASS);
    }
    if (nlhs > 5){
        plhs[5] = resize_and_create_mxRow(Dif, *it, mxREAL_CLASS);
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    /* real type is determined by the first parameter Y if nonempty;
     * or by the sixth parameter Yl1 */
    if ((!mxIsEmpty(prhs[0]) && mxIsDouble(prhs[0])) ||
        (nrhs > 5 && !mxIsEmpty(prhs[5]) && mxIsDouble(prhs[5]))){
        check_args(nrhs, prhs, args_real_t, n_real_t, mxDOUBLE_CLASS,
            "double");
        cp_pfdr_d1_ql1b_mex<double, mxDOUBLE_CLASS>(nlhs, plhs, nrhs, prhs);
    }else{
        check_args(nrhs, prhs, args_real_t, n_real_t, mxSINGLE_CLASS,
            "single");
        cp_pfdr_d1_ql1b_mex<float, mxSINGLE_CLASS>(nlhs, plhs, nrhs, prhs);
    }
}
