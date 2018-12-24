/*=============================================================================
 * tools for manipulating matrices
 *
 * Parallel implementation with OpenMP API
 *
 * Hugo Raguet 2016, 2018
 *===========================================================================*/
#pragma once
#define FULL_ATA ((size_t) 0)

template <typename real_t>
real_t operator_norm_matrix(size_t M, size_t N, const real_t *A,
    const real_t* D = nullptr, real_t tol = 1e-3, int it_max = 100,
    int nb_init = 10, bool verbose = false);
/* compute the square operator norm of a real matrix, ||A^t A||
 * using power method
 *
 * M, N    - matrix dimensions; set M or N to zero (macro FULL_ATA) for
 *           _symmetrized_ versions, that is when argument A actually contains
 *           the product (A^t A) or (A A^t)
 * A       - if M and N are positive, A is an M-by-N array, column-major format
 *           if M or N is zero (macro FULL_ATA), then A is actually (A^t A) or
 *           (A A^t), M-by-M or N-by-N (whichever is positive) array,
 *           column-major format
 * D       - diagonal equilibration matrix, actually compute ||D A^t A D||,
 *           array of length M or N (whichever is positive)
 * tol     - stopping criterion on relative norm evolution
 * it_max  - maximum number of iterations
 * nb_init - number of random initializations
 * verbose - if true, display information */

template <typename real_t>
void symmetric_equilibration_jacobi(size_t M, size_t N, const real_t* A, real_t* D);
/* extract inverse square root of the diagonal of A^t A into array pointed by D 
 *
 * M, N - matrix dimensions; set M to zero or less for _symmetrized_ version,
 *        that is when argument A actually contains the product (A^t A)
 * A    - if M and N are nonzero, A is an M-by-N array, column-major format
 *        if M is zero (macro FULL_ATA), then A is actually (A^t A),
 *        N-by-N array, column-major format
 *
 * D    - pointer to array of length N */

template <typename real_t>
void symmetric_equilibration_bunch(size_t M, size_t N, const real_t* A, real_t* D);
/* diagonal l_inf-norm scaling of A^t A into array pointed by D
 * 
 * Reference: J. R. Bunch, Equilibration of Symmetric Matrices in the Max-Norm,
 * Journal of the ACM, 1971, 18, 566-572 */
