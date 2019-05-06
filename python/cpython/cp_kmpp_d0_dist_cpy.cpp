/*=============================================================================
 * Comp, rX, it, Obj, Time, Dif = cp_kmpp_d0_dist(
 *          loss, Y, first_edge, adj_vertices, edge_weights, vert_weights, 
 *          coor_weights, cp_dif_tol, cp_it_max, K, split_iter_num,
 *          kmpp_init_num, kmpp_iter_num, verbose, max_num_threads,
 *          balance_parallel_split)
 * 
 *  Baudoin Camille 2019
 *===========================================================================*/
#include <cstdint>
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "../../include/cp_kmpp_d0_dist.hpp"

using namespace std;

/* index_t must be able to represent the number of vertices and of (undirected)
 * edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph, as well as the dimension D */
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

/* template for handling both single and double precisions */
template<typename real_t, NPY_TYPES pyREAL_CLASS>
static PyObject* cp_kmpp_d0_dist(real_t loss, PyArrayObject* py_Y,
    PyArrayObject* py_first_edge, PyArrayObject* py_adj_vertices,
    PyArrayObject* py_edge_weights, PyArrayObject* py_vert_weights,
    PyArrayObject* py_coor_weights, real_t cp_dif_tol, int cp_it_max,
    int K, int split_iter_num, int kmpp_init_num, int kmpp_iter_num,
    int verbose, int max_num_threads, int balance_parallel_split, 
    int compute_Obj, int compute_Time, int compute_Dif)
{
    /**  get inputs  **/

    /* sizes and loss */
    npy_intp* py_Y_dims = PyArray_DIMS(py_Y);
    size_t D = PyArray_NDIM(py_Y) > 1 ? py_Y_dims[0] : 1;
    index_t V = PyArray_NDIM(py_Y) > 1 ? py_Y_dims[1] : py_Y_dims[0];

    const real_t* Y = (real_t*) PyArray_DATA(py_Y);
    const real_t* vert_weights = (PyArray_SIZE(py_vert_weights) > 0) ?
        (real_t*) PyArray_DATA(py_vert_weights) : nullptr;
    const real_t* coor_weights = (PyArray_SIZE(py_coor_weights) > 0) ?
        (real_t*) PyArray_DATA(py_coor_weights) : nullptr;

    /* graph structure */
    index_t E = PyArray_SIZE(py_adj_vertices);
    const index_t *first_edge = (index_t*) PyArray_DATA(py_first_edge);
    const index_t *adj_vertices = (index_t*) PyArray_DATA(py_adj_vertices);

    /* penalizations */
    const real_t *edge_weights = (PyArray_SIZE(py_edge_weights) > 1) ?
        (real_t*) PyArray_DATA(py_edge_weights) : nullptr;
    real_t* ptr_edge_weights = (real_t*) PyArray_DATA(py_edge_weights);
    real_t homo_edge_weight = (PyArray_SIZE(py_edge_weights) == 1) ?
        ptr_edge_weights[0] : 1;
    if (max_num_threads<=0){
        max_num_threads = omp_get_max_threads();
    }

    /**  prepare output; rX is created later  **/

    npy_intp size_py_comp_t[] = {V};
    PyArrayObject* py_comp_t = (PyArrayObject*) PyArray_Zeros(1,
        size_py_comp_t, PyArray_DescrFromType(COMP_CLASS), 1);
    comp_t* Comp = (comp_t*) PyArray_DATA(py_comp_t); 

    npy_intp size_py_it[] = {1};
    PyArrayObject* py_it = (PyArrayObject*) PyArray_Zeros(1, size_py_it,
        PyArray_DescrFromType(NPY_UINT32), 1);
    int* it = (int*) PyArray_DATA(py_it);

    real_t* Obj = nullptr;
    PyArrayObject* py_Obj = (PyArrayObject*) Py_None;
    if (compute_Obj){
        npy_intp size_py_Obj[] = {cp_it_max + 1};
        py_Obj = (PyArrayObject*) PyArray_Zeros(1, size_py_Obj,
            PyArray_DescrFromType(pyREAL_CLASS), 1);
        Obj = (real_t*) PyArray_DATA(py_Obj);
    }

    double* Time = nullptr;
    PyArrayObject* py_Time = (PyArrayObject*) Py_None;
    if (compute_Time){
        npy_intp size_py_Time[] = {cp_it_max + 1};
        py_Time = (PyArrayObject*) PyArray_Zeros(1, size_py_Time,
            PyArray_DescrFromType(NPY_FLOAT64), 1);
        Time = (double*) PyArray_DATA(py_Time);
    }

    real_t* Dif = nullptr;
    PyArrayObject *py_Dif = (PyArrayObject*) Py_None;
    if (compute_Dif){
        npy_intp size_py_Dif[] = {cp_it_max};
        py_Dif = (PyArrayObject*) PyArray_Zeros(1, size_py_Dif,
            PyArray_DescrFromType(pyREAL_CLASS), 1);
        Dif = (real_t*) PyArray_DATA(py_Dif);
    }

    /**  cut-pursuit with preconditioned forward-Douglas-Rachford  **/

    Cp_d0_dist<real_t, index_t, comp_t> *cp =
        new Cp_d0_dist<real_t, index_t, comp_t>
            (V, E, first_edge, adj_vertices, Y, D);

    cp->set_loss(loss, Y, vert_weights, coor_weights);
    cp->set_edge_weights(edge_weights, homo_edge_weight);
    cp->set_monitoring_arrays(Obj, Time, Dif);
    cp->set_components(0, Comp);
    cp->set_cp_param(cp_dif_tol, cp_it_max, verbose);
    cp->set_split_param(K, split_iter_num);
    cp->set_kmpp_param(kmpp_init_num, kmpp_iter_num);
    cp->set_parallel_param(max_num_threads, balance_parallel_split);

    *it = cp->cut_pursuit();

    /* copy reduced values */
    comp_t rV = cp->get_components();
    real_t *cp_rX = cp->get_reduced_values();
    npy_intp size_py_rX[] = {(npy_intp) D, rV};
    PyArrayObject *py_rX = (PyArrayObject*) PyArray_Zeros(2, size_py_rX,
        PyArray_DescrFromType(pyREAL_CLASS), 1);
    real_t *rX = (real_t*) PyArray_DATA(py_rX);
    for (size_t rvd = 0; rvd < rV*D; rvd++){ rX[rvd] = cp_rX[rvd]; }
    
    cp->set_components(0, nullptr); // prevent Comp to be free()'d
    delete cp;
    return Py_BuildValue("OOOOOO", py_comp_t, py_rX, py_it, py_Obj, py_Time,
        py_Dif);
}

/* My python wrapper */
static PyObject* cp_kmpp_d0_dist_cpy(PyObject* self, PyObject* args)
{ 
    /* My INPUT */
    PyArrayObject *py_Y, *py_first_edge, *py_adj_vertices, *py_edge_weights,
        *py_vert_weights, *py_coor_weights;
    double loss, cp_dif_tol;  
    int cp_it_max, K, split_iter_num, kmpp_init_num, kmpp_iter_num, verbose, 
        max_num_threads, balance_parallel_split, real_t_double, compute_Obj, 
        compute_Time, compute_Dif;

    /* parse the input, from Python Object to C PyArray, double, or int type */
#if PY_MAJOR_VERSION >= 3
    if(!PyArg_ParseTuple(args, "dOOOOOOdiiiiiipipppp", &loss, &py_Y,
#else // python 2 does not accept the 'p' format specifier
    if(!PyArg_ParseTuple(args, "dOOOOOOdiiiiiiiiiiii", &loss, &py_Y,
#endif
        &py_first_edge, &py_adj_vertices, &py_edge_weights, &py_vert_weights,
        &py_coor_weights, &cp_dif_tol, &cp_it_max, &K, &split_iter_num, 
        &kmpp_init_num, &kmpp_iter_num, &verbose, &max_num_threads, 
        &balance_parallel_split, &real_t_double, &compute_Obj, &compute_Time, 
        &compute_Dif)){
        return NULL;
    }

    if (real_t_double){ /* real_t type is double */
        PyObject* PyReturn = cp_kmpp_d0_dist<double, NPY_FLOAT64>(loss, py_Y,
            py_first_edge, py_adj_vertices, py_edge_weights, py_vert_weights,
            py_coor_weights, cp_dif_tol, cp_it_max, K, split_iter_num,
            kmpp_init_num, kmpp_iter_num, verbose, max_num_threads, 
            balance_parallel_split, compute_Obj, compute_Time, compute_Dif);
        return PyReturn;
    }else{ /* real_t type is float */
        PyObject* PyReturn = cp_kmpp_d0_dist<float, NPY_FLOAT32>(loss, py_Y,
            py_first_edge, py_adj_vertices, py_edge_weights, py_vert_weights,
            py_coor_weights, cp_dif_tol, cp_it_max, K, split_iter_num,
            kmpp_init_num, kmpp_iter_num, verbose, max_num_threads, 
            balance_parallel_split, compute_Obj, compute_Time, compute_Dif);
        return PyReturn;
    }
}

static PyMethodDef cp_kmpp_d0_dist_methods[] = {
    {"cp_kmpp_d0_dist_cpy", cp_kmpp_d0_dist_cpy, METH_VARARGS,
        "wrapper for parallel cut-pursuit d0 distance"},
    {NULL, NULL, 0, NULL}
}; 

/* module initialization */
#if PY_MAJOR_VERSION >= 3
/* Python version 3 */
static struct PyModuleDef cp_kmpp_d0_dist_module = {
    PyModuleDef_HEAD_INIT,
    "cp_kmpp_d0_dist_cpy", /* name of module */
    NULL, /* module documentation, may be null */
    -1,   /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
    cp_kmpp_d0_dist_methods, /* actual methods in the module */
    NULL, /* multi-phase initialization, may be null */
    NULL, /* traversal function, may be null */
    NULL, /* clearing function, may be null */
    NULL  /* freeing function, may be null */
};

PyMODINIT_FUNC
PyInit_cp_kmpp_d0_dist_cpy(void)
{
    import_array() /* IMPORTANT: this must be called to use numpy array */
    return PyModule_Create(&cp_kmpp_d0_dist_module);
}

#else

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC
initcp_kmpp_d0_dist_cpy(void)
{
    (void) Py_InitModule("cp_kmpp_d0_dist_cpy", cp_kmpp_d0_dist_methods);
    import_array() /* IMPORTANT: this must be called to use numpy array */
}

#endif
