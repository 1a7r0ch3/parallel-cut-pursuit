## Cut-Pursuit Algorithms, Parallelized Along Components

Generic C++ classes for implementing cut-pursuit algorithms.  
Specialization to convex problems involving graph total variation, as explained in our articles [(Landrieu and Obozinski, 2016; Raguet and Landrieu, 2018)](#references).  
Parallel implementation with OpenMP API.  
MEX API for interface with GNU Octave or Matlab.  

### General problem statement
The cut-pursuit algorithms minimize functionals structured, over a weighted graph _G_ = (_V_, _E_, _w_), as 

    _F_: _x_ ∈ ℍ<sup>_V_</sup> ↦ _f_(_x_) + 
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _w_<sub>(_u_,_v_)</sub>  _ψ_(_x_<sub>_u_</sub>, _x_<sub>_v_</sub>) ,    

where ℍ is some base space, and the functional _ψ_: ℍ² → ℝ penalizes dissimilarity between its arguments, in order to enforce solutions which are *piecewise constant along the graph _G_*.

The cut-pursuit approach is to seek partitions __*V*__ of the set of vertices _V_, constituting the constant connected components of the solution, by successively solving the corresponding problem, structured over the reduced graph __*G*__ = (__*V*__, __*E*__), that is

  arg min<sub>_ξ_ ∈ ℍ<sup>__*V*__</sup></sub>
  _F_(_x_) ,    such that ∀ _U_ ∈ __*V*__, ∀ _u_ ∈ _U_, _x_<sub>_u_</sub> = _ξ_<sub>_U_</sub> ,

and then refining the partition.  
A key requirement is thus the ability to solve the reduced problem, which often have the exact same structure as the original one, but with much less vertices |__*V*__| ≪ |_V_|. If the solution of the original problem has only few constant connected components in comparison to the number of vertices, the cut-pursuit strategy can speed-up minimization by several orders of magnitude.  

Cut-pursuit algorithms come in two main flavors, namely “directionally differentiable” and “noncontinuous”.  
* In the **directionally differentiable** case, the base space ℍ is typically a vector space, and it is required that _f_ is differentiable, or at least that its nondifferentiable part is _separable along the graph_ and admits (potentially infinite) _directional derivatives_. This comprises notably many convex problems, where 
_ψ_(_x_<sub>_u_</sub>, _x_<sub>_v_</sub>) = ║<i>x</i><sub>_u_</sub> − _x_<sub>_v_</sub>║, that is to say involving a _**graph total variation**_. The refinement of the partition is based on the search for a steep directional derivative, and the reduced problem is solved using convex or continuous optimization; optimality guarantees can be provided.  
* In the **noncontinuous** case, the dissimilarity penalization typically uses _ψ_(_x_<sub>_u_</sub>, _x_<sub>_v_</sub>) = 0 if _x_<sub>_u_</sub> =_x_<sub>_v_</sub>, 1 otherwise, resulting in a measure of the _**contour**_ of the constant connected components. The functional _f_ is typically required to be separable along the graph, and to have computational properties favorable enough for solving reduced problems. The refinement of the partition relies on greedy heuristics.  
Both flavors admit multidimensional extensions, that is to say ℍ is not required to be only scalar.

### Generic classes
The class `Cp_graph` is a modification of the `Graph` class of Y. Boykov and V. Kolmogorov, for making use of their [maximum flow algorithm](#references).  
The class `Cp` is the most generic, defining all steps of the cut-pursuit approach in virtual methods.  
The class `Cp_d1` specializes methods for directionally differentiable cases involving the graph total variation.  

### Specialization `Cp_d1_ql1b`: quadratic functional, ℓ<sub>1</sub> norm, bounds, and graph total variation
The base space is ℍ = ℝ, and the general form is  

    _F_: _x_ ∈ ℝ<sup>_V_</sup> ↦  1/2 ║<i>y</i><sup>(q)</sup> − _A_<i>x</i>║<sup>2</sup> +
 ∑<sub>_v_ ∈ _V_</sub> _λ_<sub>_v_</sub> |_y_<sup>(ℓ<sub>1</sub>)</sup> − _x_<sub>_v_</sub>| +
 ∑<sub>_v_ ∈ _V_</sub> _ι_<sub>[_m_<sub>_v_</sub>, _M_<sub>_v_</sub>]</sub>(_x_<sub>_v_</sub>) +
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _w_<sub>(_u_,_v_)</sub>
 |_x_<sub>_u_</sub> − _x_<sub>_v_</sub>| ,   

where _y_<sup>(q)</sup> ∈ ℝ<sup>_n_</sup>, 
_A_: ℝ<sup>_n_</sup> → ℝ<sup>_V_</sup> is a linear operator, 
_y_<sup>(ℓ<sub>1</sub>)</sup> ∈ ℝ<sup>_V_</sup> and 
_λ_ ∈ ℝ<sup>_V_</sup> and _w_ ∈ ℝ<sup>_E_</sup> are regularization weights, 
_m_, _M_ ∈ ℝ<sup>_V_</sup> are parameters and 
_ι_<sub>[_a_,_b_]</sub> is the convex indicator of [_a_, _b_] : x ↦ 0 if _x_ ∈ [_a_, _b_], +∞ otherwise.  

When _y_<sup>(ℓ<sub>1</sub>)</sup> is zero, the combination of ℓ<sub>1</sub> norm and total variation is sometimes coined _fused LASSO_.  

Currently, _A_ must be provided as a matrix. See the documentation for special cases.  

The reduced problem is solved using the [preconditioned forward-Douglas–Rachford splitting algorithm](https://1a7r0ch3.github.io/fdr/) (see also the [corresponding repository](https://github.com/1a7r0ch3/pcd-prox-split).  

A use case where _A_ is a full ill-conditioned matrix, with positivity and fused LASSO constraints is provided with [GNU Octave or Matlab](#gnu-octave-or-matlab) interface, on a task of _brain source identification with electroencephalography_.  

### Specialization `Cp_d1_lsx`: separable loss, simplex constraints, and graph total variation
The base space is ℍ = ℝ<sup>_K_</sup>, where _K_ is a set of labels, and the general form is  

    _F_: _x_ ∈ ℝ<sup>_V_ ⨯ _K_</sup> ↦  _f_(_y_, _x_) +
 ∑<sub>_v_ ∈ _V_</sub> _ι_<sub>Δ<sub>_K_</sub></sub>(_x_<sub>_v_</sub>) +
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _w_<sub>(_u_,_v_)</sub>
 ∑<sub>_k_ ∈ _K_</sub> _λ_<sub>_k_</sub> |_x_<sub>_u_,_k_</sub> − _x_<sub>_v_,_k_</sub>| ,  

where _y_ ∈ ℝ<sup>_V_ ⨯ _K_</sup>, _f_ is a loss functional (see below), _w_ ∈ ℝ<sup>_E_</sup> and _λ_ ∈ ℝ<sup>_K_</sup> are regularization weights, and _ι_<sub>Δ<sub>_K_</sub></sub> is the convex indicator of the simplex
Δ<sub>_K_</sub> = {_x_ ∈ ℝ<sup>_K_</sup> | ∑<sub>_k_</sub> _x_<sub>_k_</sub> = 1 and ∀ _k_, _x_<sub>_k_</sub> ≥ 0}: _x_ ↦ 0 if _x_ ∈ Δ<sub>_K_</sub>, +∞ otherwise. 

The following loss functionals are available.  
Linear: _f_(_y_, _x_) = − ∑<sub>_v_ ∈ _V_</sub> ∑<sub>_k_ ∈ _K_</sub> _x_<sub>_v_,_k_</sub> _y_<sub>_v_,_k_</sub>  
Quadratic: _f_(_y_, _x_) = ∑<sub>_v_ ∈ _V_</sub> ∑<sub>_k_ ∈ _K_</sub> (_x_<sub>_v_,_k_</sub> − _y_<sub>_v_,_k_</sub>)<sup>2</sup>  
Smoothed Kullback–Leibler divergence: _f_(_y_, _x_) = ∑<sub>_v_ ∈ _V_</sub>
KL(_α_ _u_ + (1 − _α_) _y_<sub>_v_</sub>, _α_ _u_ + (1 − _α_) _x_<sub>_v_</sub>),  
where _α_ ∈ \]0,1\[,
_u_ ∈ Δ<sub>_K_</sub> is the uniform discrete distribution over _K_,
and
KL: (_p_, _q_) ↦ ∑<sub>_k_ ∈ _K_</sub> _p_<sub>_k_</sub> log(_p_<sub>_k_</sub>/_q_<sub>_k_</sub>).  

The reduced problem is also solved using the [preconditioned forward-Douglas–Rachford splitting algorithm](https://1a7r0ch3.github.io/fdr/) (see also the [corresponding repository](https://github.com/1a7r0ch3/pcd-prox-split).  

A use case with the smoothed Kullback–Leibler is provided with [GNU Octave or Matlab](#gnu-octave-or-matlab) interface, on a task of _spatial regularization of semantic classification of a 3D point cloud_.  

### Noncontinuous specializations
For now, see the [cut-pursuit](https://github.com/loicland/cut-pursuit) repository of Loïc Landrieu.
Available soon in parallelized version!

### Directory tree
    .   
    ├── data/       various data for illustration
    ├── include/    C++ headers, with some doc  
    ├── octave/     GNU Octave or Matlab code  
    │   ├── doc/    some documentation  
    │   └── mex/    MEX API  
    └── src/        C++ sources  

### C++
The C++ classes are documented within the corresponding headers in `include/`.  

### GNU Octave or Matlab
The MEX interfaces are documented within dedicated `.m` files in `mex/doc/`.  
See `compile_mex.m` for typical compilation commands under UNIX systems.  

The script `example_EEG.m` exemplifies the use of [`Cp_d1_ql1b`](#specialization-Cp_d1_ql1b-quadratic-functional-ℓ1-norm-bounds-and-graph-total-variation), on a task of _brain source identification with electroencephalography_.  

<table><tr>
<td width="10%"></td>
<td width="20%"> ground truth </td>
<td width="10%"></td>
<td width="20%"> raw retrieved activity </td>
<td width="10%"></td>
<td width="20%"> identified sources </td>
<td width="10%"></td>
</tr><tr>
<td width="10%"></td>
<td width="20%"><img src="data/EEG_ground_truth.png" width="100%"/></td>
<td width="10%"></td>
<td width="20%"><img src="data/EEG_brain_activity.png" width="100%"/></td>
<td width="10%"></td>
<td width="20%"><img src="data/EEG_brain_sources.png" width="100%"/></td>
<td width="10%"></td>
</tr></table>

<small>Data courtesy of Ahmad Karfoul and Isabelle Merlet, LTSI, INSERM U1099.</small>  

The script `example_labeling_3D.m` exemplifies the use of [`Cp_d1_lsx`](#specialization-Cp_d1_lsx-separable-loss-simplex-constraints-and-graph-total-variation), on a task of _spatial regularization of semantic classification of a 3D point cloud_.  

<table><tr>
<td width="5%"></td>
<td width="25%"> ground truth </td>
<td width="5%"></td>
<td width="25%"> random forest classifier </td>
<td width="5%"></td>
<td width="25%"> regularized classification </td>
<td width="5%"></td>
</tr><tr>
<td width="5%"></td>
<td width="25%"><img src="data/labeling_3D_ground_truth.png" width="100%"/></td>
<td width="5%"></td>
<td width="25%"><img src="data/labeling_3D_random_forest.png" width="100%"/></td>
<td width="5%"></td>
<td width="25%"><img src="data/labeling_3D_regularized.png" width="100%"/></td>
<td width="5%"></td>
</tr></table>

## References
L. Landrieu and G. Obozinski, [Cut Pursuit: Fast Algorithms to Learn Piecewise Constant Functions on Weighted Graphs](http://epubs.siam.org/doi/abs/10.1137/17M1113436), 2017.  

H. Raguet and L. Landrieu, [Cut-pursuit Algorithm for Regularizing Nonsmooth Functionals with Graph Total Variation](https://1a7r0ch3.github.io/cp/), 2018.  

Y. Boykov and V. Kolmogorov, An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision, IEEE Transactions on Pattern Analysis and Machine Intelligence, 2004.

## License
This software is under the GPLv3 license.
