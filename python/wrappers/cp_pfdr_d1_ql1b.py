import numpy as np
import os 
import sys
from cp_pfdr_d1_ql1b_ext import cp_pfdr_d1_ql1b_ext

def cp_pfdr_d1_ql1b(Y, A, first_edge, adj_vertices, edge_weights=None, 
                    Yl1=None, l1_weights=None, low_bnd=None, upp_bnd=None, 
                    cp_dif_tol=1e-5, cp_it_max=10, pfdr_rho=1., 
                    pfdr_cond_min=1e-2, pfdr_dif_rcd=0., pfdr_dif_tol=None, 
                    pfdr_it_max=int(1e4), verbose=int(1e3), 
                    AtA_if_square=True, compute_Obj=False,
                    compute_Time=False, compute_Dif=False):
    """
    Comp, rX, cp_it, Obj, Time, Dif = cp_pfdr_d1_ql1b(
            Y | AtY, A | AtA, first_edge, adj_vertices, edge_weights=None,
            Yl1=None, l1_weights=None, low_bnd=None, upp_bnd=None,
            cp_dif_tol=1e-5, cp_it_max=10, pfdr_rho=1.0, pfdr_cond_min=1e-2,
            pfdr_dif_rcd=0.0, pfdr_dif_tol=1e-3*cp_dif_tol,
            pfdr_it_max=int(1e4), verbose=int(1e3), AtA_if_square=True,
            compute_Obj=False, compute_Time=False, compute_Dif=False)

    Cut-pursuit algorithm with d1 (total variation) penalization, with a 
    quadratic functional, l1 penalization and box constraints:

    minimize functional over a graph G = (V, E)

        F(x) = 1/2 ||y - A x||^2 + ||x||_d1 + ||yl1 - x||_l1 + i_[m,M](x)

    where y in R^N, x in R^V, A in R^{N-by-|V|}
          ||x||_d1 = sum_{uv in E} w_d1_uv |x_u - x_v|,
          ||x||_l1 = sum_{v  in V} w_l1_v |x_v|,
          and the convex indicator
          i_[m,M] = infinity if it exists v in V such that x_v < m_v or 
          x_v > M_v
                  = 0 otherwise;

    using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
    splitting algorithm.

    It is easy to introduce a SDP metric weighting the squared l2-norm
    between y and A x. Indeed, if M is the matrix of such a SDP metric,
    ||y - A x||_M^2 = ||Dy - D A x||^2, with D = M^(1/2).
    Thus, it is sufficient to call the method with Y <- Dy, and A <- D A.
    Moreover, when A is the identity and M is diagonal (weighted square l2 
    distance between x and y), one should call on the precomposed version 
    (see below) with Y <- DDy = My and A <- D2 = M.

    INPUTS: real numeric type is either float32 or float64, not both;
            indices numeric type is uint32.

    NOTA: by default, components are identified using uint16_t identifiers; 
    this can be easily changed in the wrapper source if more than 65535
    components are expected (recompilation is necessary)

    Y - observations, (real) array of length N (direct matricial case) or of 
        length V (premultiplied to the left by A^t), or empty matrix (for all 
        zeros)
    A - matrix, (real) N-by-V array (direct matricial case), or V-by-V array
        (premultiplied to the left by A^t), or V-by-1 array (square diagonal
        of A^t A = A^2), or nonzero scalar (for identity matrix), or zero 
        scalar (for no quadratic part); if N = V in a direct matricial case, 
        the last argument AtA_if_square must be set to false
    first_edge, adj_vertices - graph forward-star representation:
        edges are numeroted (C-style indexing) so that all vertices originating
        from a same vertex are consecutive;
        for each vertex, 'first_edge' indicates the first edge starting 
            from the vertex (or, if there are none, starting from the next 
            vertex); array of length V+1 (uint32), the last value is the total
            number of edges;
        for each edge, 'adj_vertices' indicates its ending vertex, array of 
            length E (uint32)
    edge_weights - array of length E or scalar for homogeneous weights (real)
    Yl1        - offset for l1 penalty, (real) array of length V,
                 or empty matrix (for all zeros)
    l1_weights - array of length V or scalar for homogeneous weights (real)
                 set to zero for no l1 penalization 
    low_bnd    - array of length V or scalar (real)
                 set to negative infinity for no lower bound
    upp_bnd    - array of length V or scalar (real)
                 set to positive infinity for no upper bound
    cp_dif_tol - stopping criterion on iterate evolution; algorithm stops if
                 relative changes (in Euclidean norm) is less than dif_tol;
                 1e-4 is a typical value; a lower one can give better precision
                 but with longer computational time and more final components
    cp_it_max  - maximum number of iterations (graph cut and subproblem)
                 10 cuts solve accurately most problems
    pfdr_rho   - relaxation parameter, 0 < rho < 2
                 1 is a conservative value; 1.5 often speeds up convergence
    pfdr_cond_min - stability of preconditioning; 0 < cond_min < 1;
                    corresponds roughly the minimum ratio to the maximum 
                    descent metric; 1e-2 is typical; a smaller value might 
                    enhance preconditioning
    pfdr_dif_rcd - reconditioning criterion on iterate evolution;
                   a reconditioning is performed if relative changes of the
                   iterate drops below dif_rcd
                   warning: reconditioning might temporarily draw minimizer 
                   away from solution, and give bad subproblem solutions
    pfdr_dif_tol - stopping criterion on iterate evolution; algorithm stops if
                   relative changes (in Euclidean norm) is less than dif_tol
                   1e-3*cp_dif_tol is a conservative value
    pfdr_it_max  - maximum number of iterations 1e4 iterations provides enough
                   precision for most subproblems
    verbose      - if nonzero, display information on the progress, every
                   'verbose' PFDR iterations
    AtA_if_square - if A is square, set this to false for direct matricial case
    compute_Obj  - compute the objective functional along iterations 
    compute_Time - monitor elapsing time along iterations
    compute_Dif  - compute relative evolution along iterations 

    OUTPUTS: Obj, Time, Dif are optional outputs, set optional input
        compute_Obj, compute_Time, compute_Dif to True to get them 

    Comp - assignement of each vertex to a component, array of length V
        (uint16)
    rX - values of each component of the minimizer, array of length rV (real);
        the actual minimizer is then reconstructed as X = rX[Comp];
    cp_it - actual number of cut-pursuit iterations performed
    Obj - if requested, the values of the objective functional along iterations
        (array of length cp_it + 1); in the precomputed A^t A version, a
        constant 1/2||Y||^2 in the quadratic part is omited
    Time - if requested, the elapsed time along iterations
        (array of length cp_it + 1)
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
    # real type is determined by the first parameter Y if nonempty; 
    # or by the second parameter A is nonscalar;
    # or by the parameter Yl1
    if type(Y) == np.ndarray and Y.any():
        real_t = Y.dtype
    elif type(Yl1) == np.ndarray and Yl1.any():
        real_t = Yl1.dtype
    else:
        raise TypeError(("At least one of arguments 'Y' or 'Yl1' "
                         "must be provided as a nonempty numpy array."))

    if real_t not in ["float32", "float64"]:
        raise TypeError(("Currently, the real numeric type must be float32 or"
                         " float64."))

    # Convert in numpy array scalar entry: Y, A, first_edge, adj_vertices, 
    # edge_weights, Yl1, l1_weights, low_bnd, upp_bnd, and define float numpy 
    # array argument with the right float type, if empty:
    if type(Y) != np.ndarray:
        raise TypeError("Argument 'Y' must be a (possibly empty) numpy array.")

    if type(A) != np.ndarray:
        if type(A) == list:
            raise TypeError("Argument 'A' must be a scalar or a numpy array.")
        else:
            A = np.array([A], real_t)

    if type(first_edge) != np.ndarray or first_edge.dtype != "uint32":
        raise TypeError(("Argument 'first_edge' must be a numpy array of "
                         "type uint32."))

    if type(adj_vertices) != np.ndarray or adj_vertices.dtype != "uint32":
        raise TypeError(("Argument 'adj_vertices' must be a numpy array of "
                         "type uint32."))

    if type(edge_weights) != np.ndarray:
        if type(edge_weights) == list:
            raise TypeError("Argument 'edge_weights' must be a scalar or a "
                            "numpy array.")
        elif edge_weights != None:
            edge_weights = np.array([edge_weights], dtype=real_t)
        else:
            edge_weights = np.array([1.0], dtype=real_t)

    if type(Yl1) != np.ndarray:
        if type(Yl1) == list:
            raise TypeError("Argument 'Yl1' must be a scalar or a numpy "
                            "array.")
        elif Yl1 != None:
            Yl1 = np.array([Yl1], dtype=real_t)
        else:
            Yl1 = np.array([], dtype=real_t)

    if type(l1_weights) != np.ndarray:
        if type(l1_weights) == list:
            raise TypeError("Argument 'l1_weights' must be a scalar or a numpy"
                            " array.")
        elif l1_weights != None:
            l1_weights = np.array([l1_weights], dtype=real_t)
        else:
            l1_weights = np.array([0.0], dtype=real_t)

    if type(low_bnd) != np.ndarray:
        if type(low_bnd) == list:
            raise TypeError("Argument 'low_bnd' must be a scalar or a numpy "
                            "array.")
        elif low_bnd != None:
            low_bnd = np.array([low_bnd], dtype=real_t)
        else: 
            low_bnd = np.array([-np.inf], dtype=real_t)

    if type(upp_bnd) != np.ndarray:
        if type(upp_bnd) == list:
            raise TypeError("Argument 'upp_bnd' must be a scalar or a numpy "
                            "array.")
        elif upp_bnd != None:
            upp_bnd = np.array([upp_bnd], dtype=real_t)
        else: 
            upp_bnd = np.array([np.inf], dtype=real_t)

    
    # Check type of all numpy.array arguments of type float (Y, A, 
    # edge_weights, Yl1, l1_weights, low_bnd, upp_bnd) 
    for name, ar_args in zip(
            ["Y", "A", "edge_weights", "Yl1", "l1_weights", "low_bnd",
             "upp_bnd"],
            [Y, A, edge_weights, Yl1, l1_weights, low_bnd, upp_bnd]):
        if ar_args.dtype != real_t:
            raise TypeError("argument '{0}' must be of type '{1}'"
                            .format(name, real_t))

    # Check fortran continuity of all numpy.array arguments of type float 
    # (Y, A, first_edge, adj_vertices, edge_weights, Yl1, l1_weights, low_bnd,
    # upp_bnd) 
    for name, ar_args in zip(
            ["Y", "A", "first_edge", "adj_vertices", "edge_weights", "Yl1",
             "l1_weights", "low_bnd", "upp_bnd"],
            [Y, A, first_edge, adj_vertices, edge_weights, Yl1, l1_weights,
             low_bnd, upp_bnd]):
        if not(ar_args.flags["F_CONTIGUOUS"]):
            raise TypeError("argument '{0}' must be F_CONTIGUOUS".format(name))

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
            ["AtA_if_square", "compute_Obj", "compute_Time", "compute_Dif"],
            [AtA_if_square, compute_Obj, compute_Time, compute_Dif]):
        if type(b_args) != bool:
            raise TypeError("argument '{0}' must be boolean".format(name))

    print(type(Y), type(A), type(first_edge), type(adj_vertices), type(edge_weights), type(Yl1), type(l1_weights), type(low_bnd), type(upp_bnd), type(cp_dif_tol), type(cp_it_max), type(pfdr_rho), type(pfdr_cond_min), type(pfdr_dif_rcd), type(pfdr_dif_tol), type(pfdr_it_max), type(verbose), type(AtA_if_square), type(real_t == "float64"), type(compute_Obj), type(compute_Time), type(compute_Dif)) 

    # Call wrapper python in C  
    Comp, rX, it, Obj, Time, Dif = cp_pfdr_d1_ql1b_ext(
            Y, A, first_edge, adj_vertices, edge_weights, Yl1, l1_weights,
            low_bnd, upp_bnd, cp_dif_tol, cp_it_max, pfdr_rho, pfdr_cond_min,
            pfdr_dif_rcd, pfdr_dif_tol, pfdr_it_max, verbose, AtA_if_square,
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


