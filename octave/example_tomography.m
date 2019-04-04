       %------------------------------------------------------------%
       %  script for testing cp_pfdr_d1_ql1b on tomography problem  %
       %------------------------------------------------------------%
% Reference: H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
% Nonsmooth Functionals with Graph Total Variation, International Conference on
% Machine Learning, PMLR, 2018, 80, 4244-4253
%
% Hugo Raguet 2017, 2018
cd(fileparts(which('example_tomography.m')));
addpath(genpath('./bin/'));

%%%  general parameters  %%%
printResults = true; % requires color encapsulated postscript driver on your

%%%  parameters; see octave/doc/cp_pfdr_d1_ql1b_mex.m %%%
cp_dif_tol = 1e-3;
cp_it_max = 10;
pfdr_rho = 1.5;
pfdr_cond_min = 1e-3;
pfdr_dif_rcd = 0;
pfdr_dif_tol = 1e-1*cp_dif_tol;
pfdr_it_max = 1e4;
pfdr_verbose = 1e3;

%%%  initialize data  %%%
% Simulated tomography: Shepp-Logan phantom 64x64 with 7 projections;
% TV Graph connectivity is about 3 pixel radius;
% Penalization parameters computed with SURE methods, heuristics adapted from
% H. Raguet: A Signal Processing Approach to Voltage-Sensitive Dye Optical
% Imaging, Ph.D. Thesis, Paris-Dauphine University, 2014
load('../data/tomography.mat')

tic;
Yl1 = []; low_bnd = 0.0; upp_bnd = 1.0;
[cv, rx] = cp_pfdr_d1_ql1b_mex(y, A, first_edge, ...
    adj_vertices, d1_weights, Yl1, l1_weights, low_bnd, upp_bnd, ...
    cp_dif_tol, cp_it_max, pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, ...
    pfdr_dif_tol, pfdr_it_max, pfdr_verbose);
time = toc;
x = rx(cv+1); % rx is components values, cv is components indices
clear cv rx;

fprintf('Total MEX execution time %.1f s\n\n', time);

%%% plot and print results  %%%
figure(1), clf, colormap('gray');
imagesc(x0); axis image; set(gca, 'Xtick', [], 'Ytick', []);
title('ground truth');
if printResults
    fprintf('print ground truth... ');
    print(gcf, '-depsc', 'tomography_ground_truth');
    fprintf('done.\n');
end

figure(2), clf, colormap('gray');
imagesc(reshape(x, size(x0))); axis image; set(gca, 'Xtick', [], 'Ytick', []);
title('reconstruction');
if printResults
    fprintf('print reconstruction... ');
    print(gcf, '-depsc', 'tomography_reconstruction');
    fprintf('done.\n');
end
