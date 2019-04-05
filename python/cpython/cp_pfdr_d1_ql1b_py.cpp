/*=============================================================================
 * (Comp, rX, it, Obj, Time, Dif) = cp_pfdr_d1_ql1b_py(Y, A , first_edge, 
 *          adj_vertices, edge_weights, Yl1, l1_weights, low_bnd, upp_bnd, 
 *          cp_dif_tol, cp_it_max, pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, 
 *          pfdr_dif_tol, pfdr_it_max, verbose, AtA_if_square)
 * 
 *  Baudoin Camille 2019
 *===========================================================================*/
#include <cstdint>
#include <string>
#include <sstream> 
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "../../include/cp_pfdr_d1_ql1b.hpp" 

using namespace std;

/* index_t must be able to represent the number of vertices and of (undirected)
 * edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph */
typedef uint32_t index_t;
# define VERTEX_CLASS NPY_UINT32 
# define VERTEX_ID "uint32"
typedef uint16_t comp_t;
# define COMP_CLASS NPY_UINT16 
# define COMP_ID "uint16"
/* uncomment the following if more than 65535 components are expected */
// typedef uint32_t comp_t;
// # define COMP_CLASS NPY_UINT32 
// # define COMP_ID "uint32"

/* arrays with arguments type */
static const int args_real_t[] = {0, 1, 4, 5, 6};
static const int n_real_t = 4;
static const int args_index_t[] = {2, 3};
static const int n_index_t = 2;

/* template for handling both single and double precisions */
template<typename real_t, NPY_TYPES pyREAL_CLASS>
static PyObject* cp_pfdr_d1_ql1b_py(PyArrayObject* py_Y,
    PyArrayObject* py_A, PyArrayObject* py_first_edge,
    PyArrayObject* py_adj_vertices, PyArrayObject* py_edge_weights,
    PyArrayObject* py_Yl1, PyArrayObject* py_l1_weights,
    PyArrayObject* py_low_bnd, PyArrayObject* py_upp_bnd, real_t cp_dif_tol,
    int cp_it_max, real_t pfdr_rho, real_t pfdr_cond_min, real_t pfdr_dif_rcd,
    real_t pfdr_dif_tol, int pfdr_it_max, int verbose, int py_AtA_if_square,
    int compute_Obj, int compute_Time, int compute_Dif)
{
    /**  get inputs  **/

    /* quadratic functional */
    npy_intp * py_A_size = PyArray_DIMS(py_A);
    size_t N = py_A_size[0];
    index_t V = py_A_size[1];

    const real_t *Y = PyArray_SIZE(py_Y) > 0 ?
        (real_t*) PyArray_DATA(py_Y) : nullptr;
    const real_t *A = (N == 1 && V == 1) ?
        nullptr : (real_t*) PyArray_DATA(py_A); 
    real_t * ptr_A = (real_t*) PyArray_DATA(py_A);
    const real_t a = (N == 1 && V == 1) ?
        ptr_A[0] : 1.0; 

    if (V == 1){ /* quadratic functional is only weighted square difference */
        if (N == 1){
            if (PyArray_SIZE(py_Y) > 0){ /* fidelity is square l2 */
                V = PyArray_SIZE(py_Y);
            }else if (PyArray_SIZE (py_edge_weights) > 0){
                /* fidelity is only l1 */
                V = PyArray_SIZE(py_edge_weights);
            }else{ /*should not happen */
                PyErr_SetString(PyExc_ValueError, "Cut-pursuit d1 quadratic l1"
                    " bounds: arguments Y and Yl1 cannot be both empty.");
            }
        }else{ /* A is given V-by-1, representing a diagonal V-by-V */
            V = N;
        }
        N = DIAG_ATA; /* DIAG_ATA is a macro */
    }else if (V == N && (py_AtA_if_square==1)){
        N = FULL_ATA; 
    }

    /* graph structure */
    index_t E = PyArray_SIZE(py_adj_vertices);
    const index_t *first_edge = (index_t*) PyArray_DATA(py_first_edge); 
    const index_t *adj_vertices = (index_t*) PyArray_DATA(py_adj_vertices); 
    if (PyArray_SIZE(py_first_edge) != (V + 1)){
        std::stringstream py_err_msg;
        py_err_msg << "Cut-pursuit d1 quadratic l1 bounds: argument 3 "
            "'adj_vertices' should contain" << (V+1) << " (|V|+1) elements, "
            "but " << PyArray_SIZE(py_first_edge) << "are given.";
        PyErr_SetString(PyExc_ValueError, py_err_msg.str().c_str());
    }

    /* penalizations */
    const real_t *edge_weights = (PyArray_SIZE(py_edge_weights) > 1) ?
        (real_t*) PyArray_DATA(py_edge_weights) : nullptr; 
    real_t * ptr_edge_weights = (real_t*) PyArray_DATA(py_edge_weights);
    real_t homo_edge_weight = (PyArray_SIZE(py_edge_weights) == 1) ?
        ptr_edge_weights[0] : 1 ;

    const real_t* Yl1 = (PyArray_SIZE(py_Yl1)>0) ? 
        (real_t*) PyArray_DATA(py_Yl1) : nullptr; 

    const real_t *l1_weights = (PyArray_SIZE(py_l1_weights) > 1) ?
        (real_t*) PyArray_DATA(py_l1_weights) : nullptr;
    real_t * ptr_l1_weights = (real_t*) PyArray_DATA(py_l1_weights);
    real_t homo_l1_weight =  (PyArray_SIZE(py_l1_weights) == 1) ?
        ptr_l1_weights[0] : 0;

    const real_t *low_bnd = (PyArray_SIZE(py_low_bnd) > 1) ?
        (real_t*) PyArray_DATA(py_low_bnd) : nullptr; 
    real_t * ptr_low_bnd = (real_t*) PyArray_DATA(py_low_bnd);
    real_t homo_low_bnd = (PyArray_SIZE(py_low_bnd) == 1) ?
        ptr_low_bnd[0] : -INF_REAL;

    const real_t *upp_bnd = (PyArray_SIZE(py_upp_bnd) > 1) ?
        (real_t*) PyArray_DATA(py_upp_bnd) : nullptr; 
    real_t * ptr_upp_bnd = (real_t*) PyArray_DATA(py_upp_bnd);
    real_t homo_upp_bnd = (PyArray_SIZE(py_upp_bnd) == 1) ?
        ptr_upp_bnd[0] : INF_REAL;

    /**  prepare output; rX is created later  **/

    npy_intp size_py_comp_t[] = {V};
    PyArrayObject* py_comp_t = (PyArrayObject*) PyArray_Zeros(1,
        size_py_comp_t, PyArray_DescrFromType(COMP_CLASS), 1);
    comp_t *Comp = (comp_t*) PyArray_DATA(py_comp_t); 

    npy_intp size_py_it[] = {1};
    PyArrayObject* py_it = (PyArrayObject*) PyArray_Zeros(1, size_py_it,
        PyArray_DescrFromType(NPY_UINT32), 1);
    int* it = (int*) PyArray_DATA(py_it); 

    real_t* Obj = nullptr;
    PyArrayObject* py_Obj = (PyArrayObject*) Py_None;
    if (compute_Obj){
        npy_intp size_py_Obj[] = {cp_it_max+1};
        py_Obj = (PyArrayObject*) PyArray_Zeros(1, size_py_Obj,
            PyArray_DescrFromType(pyREAL_CLASS), 1);
        Obj = (real_t*) PyArray_DATA(py_Obj);
    }

    double* Time = nullptr;
    PyArrayObject* py_Time = (PyArrayObject*) Py_None;
    if (compute_Time){
        npy_intp size_py_Time[] = {cp_it_max+1};
        py_Time = (PyArrayObject*) PyArray_Zeros(1, size_py_Time,
            PyArray_DescrFromType(pyREAL_CLASS), 1);
        Time = (double*) PyArray_DATA(py_Time);
    }

    real_t* Dif = nullptr;
    PyArrayObject* py_Dif = (PyArrayObject*) Py_None;
    if (compute_Dif){
        npy_intp size_py_Dif[] = {cp_it_max};
        py_Dif = (PyArrayObject*) PyArray_Zeros(1, size_py_Dif,
            PyArray_DescrFromType(pyREAL_CLASS), 1);
        Dif = (real_t*) PyArray_DATA(py_Dif);
    }

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
    real_t* cp_rX = cp->get_reduced_values();
    npy_intp size_py_rX[] = {rV};
    PyArrayObject* py_rX = (PyArrayObject*) PyArray_Zeros(1, size_py_rX,
        PyArray_DescrFromType(pyREAL_CLASS), 1);
    real_t *rX = (real_t*) PyArray_DATA(py_rX);
    for (comp_t rv = 0; rv < rV; rv++){ rX[rv] = cp_rX[rv]; }

    cp->set_components(0, nullptr); // prevent Comp to be free()'d
    delete cp;
    return Py_BuildValue("OOOOOO", py_comp_t, py_rX, py_it, py_Obj, py_Time,
        py_Dif); 
}

/* My python wrapper */
static PyObject* py_C_API_function(PyObject * self, PyObject * args)
{ 
    /* My INPUT */ 
    PyArrayObject *py_Y, *py_A, *py_first_edge, *py_adj_vertices,
        *py_edge_weights, *py_Yl1, *py_l1_weights, *py_low_bnd, *py_upp_bnd; 
    double cp_dif_tol, pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol;
    int cp_it_max, pfdr_it_max, verbose, AtA_if_square, real_t_double,
        compute_Obj, compute_Time, compute_Dif; 
    
    /* parse the input, from python Object to c PyArray, double, or int type */
    if(!PyArg_ParseTuple(args, "OOOOOOOOOdiddddiippppp", &py_Y, &py_A,
        &py_first_edge, &py_adj_vertices, &py_edge_weights, &py_Yl1,
        &py_l1_weights, &py_low_bnd, &py_upp_bnd, &cp_dif_tol, &cp_it_max,
        &pfdr_rho, &pfdr_cond_min, &pfdr_dif_rcd, &pfdr_dif_tol, &pfdr_it_max,
        &verbose, &AtA_if_square, &real_t_double, &compute_Obj, &compute_Time,
        &compute_Dif)) {
        return NULL;
    }

    if (real_t_double){ /* real_t type is double */
        PyObject* PyReturn = cp_pfdr_d1_ql1b_py<double, NPY_FLOAT64>(py_Y,
            py_A, py_first_edge, py_adj_vertices, py_edge_weights, py_Yl1,
            py_l1_weights, py_low_bnd, py_upp_bnd, cp_dif_tol, cp_it_max,
            pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol, pfdr_it_max,
            verbose, AtA_if_square, compute_Obj, compute_Time, compute_Dif);
        return PyReturn;
    }else{ /* real_t type is float */
        float cp_dif_tol_f = (float) cp_dif_tol;
        float pfdr_rho_f = (float) pfdr_rho;
        float pfdr_cond_min_f = (float) pfdr_cond_min;
        float pfdr_dif_rcd_f = (float) pfdr_dif_rcd;
        float pfdr_dif_tol_f = (float) pfdr_dif_tol;
        PyObject* PyReturn = cp_pfdr_d1_ql1b_py<float, NPY_FLOAT32>(py_Y, py_A,
            py_first_edge, py_adj_vertices, py_edge_weights, py_Yl1,
            py_l1_weights, py_low_bnd, py_upp_bnd, cp_dif_tol_f, cp_it_max,
            pfdr_rho_f, pfdr_cond_min_f, pfdr_dif_rcd_f, pfdr_dif_tol_f,
            pfdr_it_max, verbose, AtA_if_square, compute_Obj, compute_Time,
            compute_Dif);
        return PyReturn;
    }
}

static PyMethodDef cp_pfdr_d1_ql1b_py_C_API_methods[] = {
    {"py_C_API_function", py_C_API_function, METH_VARARGS,
        "wrapper for parallel cut pursuit"},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
/* module initialization */
/* Python version 3 */
static struct PyModuleDef cp_pfdr_d1_ql1b_py_C_API_module = {
    PyModuleDef_HEAD_INIT,
    "cp_pfdr_d1_ql1b_py_C_API",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    cp_pfdr_d1_ql1b_py_C_API_methods
};

PyMODINIT_FUNC
PyInit_cp_pfdr_d1_ql1b_py_C_API(void)
{
    import_array() /* IMPORTANT: this must be called to use numpy array */
    return PyModule_Create(&cp_pfdr_d1_ql1b_py_C_API_module);
}

#else

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC
initcp_pfdr_d1_ql1b_py_C_API_module(void)
{
    (void) Py_InitModule("cp_pfdr_d1_ql1b_py_C_API",
        cp_pfdr_d1_ql1b_py_C_API_methods);
    import_array() /* IMPORTANT: this must be called to use numpy array */
}

#endif

