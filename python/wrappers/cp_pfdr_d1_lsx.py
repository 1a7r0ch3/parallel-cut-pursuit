import numpy as np
import os 
import sys
from cp_pfdr_d1_lsx_ext import cp_pfdr_d1_lsx_ext

def cp_pfdr_d1_lsx(loss, Y, first_edge, adj_vertices, edge_weights=None, 
                      loss_weights=None, d1_coor_weights=None, cp_dif_tol=1e-3,
                      cp_it_max=10, pfdr_rho=1., pfdr_cond_min=1e-2, 
                      pfdr_dif_rcd=0., pfdr_dif_tol=None, pfdr_it_max=int(1e4),
                      verbose=int(1e2), compute_Obj=False, compute_Time=False, 
                      compute_Dif=False):

    """
    Comp, rX, cp_it, Obj, Time, Dif = cp_pfdr_d1_lsx(
            loss, Y, first_edge, adj_vertices, edge_weights=None,
            loss_weights=None, d1_coor_weights=None, cp_dif_tol=1e-3,
            cp_it_max=10, pfdr_rho=1.0, pfdr_cond_min=1e-2, pfdr_dif_rcd=0.0,
            pfdr_dif_tol=1e-3*cp_dif_tol, pfdr_it_max=1e4, verbose=1e2)

    Cut-pursuit algorithm with d1 (total variation) penalization, with a 
    separable loss term and simplex constraints:

    minimize functional over a graph G = (V, E)

        F(x) = f(x) + ||x||_d1 + i_{simplex}(x)

    where for each vertex, x_v is a D-dimensional vector,
          f is a separable data-fidelity loss
          ||x||_d1 = sum_{uv in E} w_d1_uv (sum_d w_d1_d |x_ud - x_vd|),
    and i_{simplex} is the standard D-simplex constraint over each vertex,
        i_{simplex} = 0 for all v, (for all d, x_vd >= 0) and sum_d x_vd = 1,
                    = infinity otherwise;

    using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
    splitting algorithm.

    Available separable data-fidelity loss include:

    linear
        f(x) = - <x, y> ,  with  <x, y> = sum_{v,d} x_{v,d} y_{v,d} ;

    quadratic
        f(x) = 1/2 ||y - x||_{l2,w}^2 ,
    with  ||y - x||_{l2,w}^2 = sum_{v,d} w_v (y_{v,d} - x_{v,d})^2 ;

    smoothed Kullback-Leibler divergence (cross-entropy)
        f(x) = sum_v w_v KLs(x_v, y_v),
    with KLs(y_v, x_v) = KL(s u + (1 - s) y_v ,  s u + (1 - s) x_v), where
        KL is the regular Kullback-Leibler divergence,
        u is the uniform discrete distribution over {1,...,D}, and
        s = loss is the smoothing parameter ;
    it yields
        KLs(y_v, x_v) = - H(s u + (1 - s) y_v)
            - sum_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) x_{v,d}) ,
    where H is the entropy, that is H(s u + (1 - s) y_v)
          = - sum_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) y_{v,d}) ;
    note that the choosen order of the arguments in the Kullback-Leibler
    does not favor the entropy of x (H(s u + (1 - s) y_v) is a constant),
    hence this loss is actually equivalent to cross-entropy.

    INPUTS: real numeric type is either float32 or float64, not both;

    NOTA: by default, components are identified using uint16_t identifiers; 
    this can be easily changed in the wrapper source if more than 65535
    components are expected (recompilation is necessary)

    loss - 0 for linear, 1 for quadratic, 0 < loss < 1 for smoothed 
        Kullback-Leibler (see above)
    Y - observations, (real) D-by-V array, column-major format (at each
        vertex, supposed to lie on the probability simplex)
    first_edge, adj_vertices - graph forward-star representation:
        edges are numeroted (C-style indexing) so that all vertices originating
            from a same vertex are consecutive;
        for each vertex, 'first_edge' indicates the first edge starting from 
            the vertex (or, if there are none, starting from the next vertex);
            array of length V+1 (uint32), the last value is the total number of
            edges;
        for each edge, 'adj_vertices' indicates its ending vertex, array of 
            length E (uint32)
    edge_weights - (real) array of length E or scalar for homogeneous weights
    loss_weights - weights on vertices; (real) array of length V or empty for
        no weights
    d1_coor_weights - for multidimensional data, weights the coordinates in the
        l1 norms of finite differences; all weights must be strictly positive,
        it is advised to normalize the weights so that the first value is unity
    cp_dif_tol - stopping criterion on iterate evolution; algorithm stops if
        relative changes (in Euclidean norm) is less than dif_tol;
        1e-3 is a typical value; a lower one can give better precision
        but with longer computational time and more final components
    cp_it_max - maximum number of iterations (graph cut and subproblem)
        10 cuts solve accurately most problems
    pfdr_rho - relaxation parameter, 0 < rho < 2
        1 is a conservative value; 1.5 often speeds up convergence
    pfdr_cond_min - stability of preconditioning; 0 < cond_min < 1;
        corresponds roughly the minimum ratio to the maximum descent metric;
        1e-2 is typical; a smaller value might enhance preconditioning
    pfdr_dif_rcd - reconditioning criterion on iterate evolution;
        a reconditioning is performed if relative changes of the iterate drops
        below dif_rcd;
        warning: reconditioning might temporarily draw minimizer away from
        solution, and give bad subproblem solutions
    pfdr_dif_tol - stopping criterion on iterate evolution; algorithm stops if
        relative changes (in Euclidean norm) is less than dif_tol
        1e-3*cp_dif_tol is a conservative value
    pfdr_it_max - maximum number of iterations
        1e4 iterations provides enough precision for most subproblems
    verbose - if nonzero, display information on the progress, every 'verbose'
        PFDR iterations
    compute_Obj  - compute the objective functional along iterations 
    compute_Time - monitor elapsing time along iterations
    compute_Dif  - compute relative evolution along iterations 

    OUTPUTS:

    Comp - assignement of each vertex to a component, array of length V
        (uint16)
    rX - values of each component of the minimizer, array of length rV (real);
        the actual minimizer is then reconstructed as X = rX[Comp];
    cp_it - actual number of cut-pursuit iterations performed
    Obj - if requested ,the values of the objective functional along iterations
        (array of length cp_it + 1)
    Time - if requested, the elapsed time along iterations (array of length
        cp_it + 1)
    Dif  - if requested, the iterate evolution along iterations
        (array of length cp_it)
     
    Parallel implementation with OpenMP API.

    H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing Nonsmooth
    Functionals with Graph Total Variation, International Conference on Machine
    Learning, PMLR, 2018, 80, 4244-4253

    H. Raguet, A Note on the Forward-Douglas--Rachford Splitting for Monotone 
    Inclusion and Convex Optimization Optimization Letters, 2018, 1-24

    Baudoin Camille 2019
    """
    
    # Determine the type of float argument (real_t) 
    # real_t type is determined by the first parameter Y 
    if Y.any() and Y.dtype == "float64":
        real_t = "float64" 
    elif Y.any() and Y.dtype == "float32":
        real_t = "float32" 
    else:
        raise TypeError(("argument 'Y' must be a non empty numpy array of "
                      "floats (float32 or float64)")) 
    
    # Convert in numpy array scalar entry: Y, first_edge, adj_vertices, 
    # edge_weights, loss_weights, d1_coor_weights and define float numpy array
    # argument with the right float type, if empty:
    if type(Y) != np.ndarray:
        raise TypeError(("argument 'Y' must be a numpy array of floats "
                      "(float32 or float64)")) 

    if type(first_edge) != np.ndarray or first_edge.dtype != "uint32":
        raise TypeError(("argument 'first_edge' must be a numpy array of "
                       "uint32"))

    if type(adj_vertices) != np.ndarray or adj_vertices.dtype != "uint32":
        raise TypeError(("argument 'adj_vertices' must be a numpy array of "
                       "uint32"))

    if type(edge_weights) != np.ndarray:
        if type(edge_weights) == list:
            raise TypeError(("argument 'edge_weights' can not be a list, must "
                        "be either a {0} or a numpy array").format(real_t))
        elif edge_weights != None:
            edge_weights = np.array([edge_weights], dtype=real_t)
        else:
            edge_weights = np.array([1.0], dtype=real_t)
        
    if type(loss_weights) != np.ndarray:
        if type(loss_weights) == list:
            raise TypeError(("argument 'loss_weights' can not be a list, must "
                        "be either a {0} or a numpy array").format(real_t))
        elif loss_weights != None:
            loss_weights = np.array([loss_weights], dtype=real_t)
        else:
            loss_weights = np.array([], dtype=real_t)

    if type(d1_coor_weights) != np.ndarray:
        if type(d1_coor_weights) == list:
            raise TypeError(("argument 'd1_coor_weights' can not be a list, "
                        "must be either a {0} or a "
                        "numpy array").format(real_t))
        elif d1_coor_weights != None:
            d1_coor_weights = np.array([d1_coor_weights], dtype=real_t)
        else:
            d1_coor_weights = np.array([], dtype=real_t)
 
    # Check type of all numpy.array arguments of type float (Y, edge_weights,
    # loss_weights, d1_coor_weights) 
    for name, ar_args in zip(
            ["Y", "edge_weights", "loss_weights", "d1_coor_weights"],
            [Y, edge_weights, loss_weights, d1_coor_weights]):
        if ar_args.dtype != real_t:
            raise TypeError("{0} must be of {1} type".format(name, real_t))

    # Check fortran continuity of all numpy.array arguments of type float (Y,
    # first_edge, adj_vertices, edge_weights, loss_weights, d1_coor_weights)
    for name, ar_args in zip(
            ["Y", "first_edge", "adj_vertices", "edge_weights", 
             "loss_weights", "d1_coor_weights"],
            [Y, first_edge, adj_vertices, edge_weights, loss_weights,
             d1_coor_weights]):
        if not(ar_args.flags["F_CONTIGUOUS"]):
            raise TypeError("{0} must be F_CONTIGUOUS".format(name))

    # Convert in float64 all float arguments if needed (cp_dif_tol, pfdr_rho,
    # pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol) 
    if pfdr_dif_tol is None:
        pfdr_dif_tol = cp_dif_tol*1e-3
    cp_dif_tol = float(cp_dif_tol)
    pfdr_rho = float(pfdr_rho)
    pfdr_cond_min = float(pfdr_cond_min)
    pfdr_dif_rcd = float(pfdr_dif_rcd)
    pfdr_dif_tol = float(pfdr_dif_tol)
     
    # Convert all int arguments (cp_it_max, pfdr_it_max, verbose) in ints: 
    cp_it_max = int(cp_it_max)
    pfdr_it_max = int(pfdr_it_max)
    verbose = int(verbose)

    # Check type of all booleen arguments (AtA_if_square, compute_Obj, 
    # compute_Time, compute_Dif)
    for name, b_args in zip(
        ["compute_Obj", "compute_Time", "compute_Dif"],
        [compute_Obj, compute_Time, compute_Dif]):
        if type(b_args) != bool:
            raise TypeError("{0} must be a boolean".format(name))

    # Call wrapper python in C  
    Comp, rX, it, Obj, Time, Dif = cp_pfdr_d1_lsx_ext(
            loss, Y, first_edge, adj_vertices, edge_weights, loss_weights,
            d1_coor_weights, cp_dif_tol, cp_it_max, pfdr_rho, pfdr_cond_min,
            pfdr_dif_rcd, pfdr_dif_tol, pfdr_it_max, verbose,
            real_t == "float64", compute_Obj, compute_Time, compute_Dif) 

    it = it[0]
    
    # Return output depending of the optional output needed
    if (compute_Obj and compute_Time and compute_Dif):
        return Comp, rX, it, Obj, Time, Dif
    elif (compute_Obj and compute_Time):
        return Comp, rX, it, Obj, Time
    elif (compute_Obj and compute_Dif):
        return Comp, rX, it, Obj, Dif
    elif (compute_Time and compute_Dif):
        return Comp, rX, it, Time, Dif
    elif (compute_Obj):
        return Comp, rX, it, Obj
    elif (compute_Time):
        return Comp, rX, it, Time
    elif (compute_Dif):
        return Comp, rX, it, Dif
    else:
        return Comp, rX, it
