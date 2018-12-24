/*=============================================================================
 * Orthogonal projection over the simplex:
 *
 *      for all d, x_d >= 0; and sum_d x_d = a
 *      
 * possibly within a diagonal metric defined by 1./m as,
 *
 *      <x, y>_{1/m} = <x, diag(1./m) y> = sum_d x_d y_d / m_d.
 *
 * i.e. m is the vector of the /inverses/ of the diagonal entries of the 
 * matrix of the desired metric (set to null for usual Euclidean metric)
 * Work in-place; parallel implementation with OpenMP API
 * 
 * Hugo Raguet 2016, 2018
 *===========================================================================*/
#pragma once
#include <cstddef>

template <typename real_t>
void proj_simplex(real_t *X, size_t D, size_t N = 1, const real_t *A = nullptr,
    real_t a = 1., const real_t *M = nullptr, const real_t *m = nullptr);
/* 7 arguments
 * X - array of N D-dimensionnal vectors, D-by-N array, column major format
 * D - ambiant dimensionality
 * N - number of input vectors
 * A - total sums of the simplices (unity yields the standard simplex)
 *     different for each input vector (array of length N)
 *     otherwise set to null and specify the common sum in 'a'
 * a - total sums of the simplices (unity yields the standard simplex)
 * M - (inverse terms of) a diagonal metric
 *     different for each input vector (D-by-N array, column major format)
 *     otherwise set to null and specify the common metric in 'm'
 * m - (inverse terms of) a diagonal metric
 *     set to null for usual euclidean metric */
