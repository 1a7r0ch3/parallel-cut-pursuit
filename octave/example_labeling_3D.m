  %------------------------------------------------------------------------%
  %  script for illustrating cp_pfdr_d1_lsx on labeling of 3D point cloud  %
  %------------------------------------------------------------------------%
% Reference: H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
% Nonsmooth Functionals with Graph Total Variation, International Conference on
% Machine Learning, PMLR, 2018, 80, 4244-4253
%
% Hugo Raguet 2017, 2018
cd(fileparts(which('example_labeling_3D.m')));
addpath(genpath('./bin/'));

%%%  classes involved in the task  %%%
classNames = {'road', 'vegetation', 'facade', 'hardscape', 'scanning artifacts', 'cars'};
classId = uint8(1:6)';

%%%  parameters; see octave/doc/cp_pfdr_d1_lsx_mex.m  %%%
CP_difTol = 1e-3;
CP_itMax = 10;
PFDR_rho = 1.5;
PFDR_condMin = 1e-2;
PFDR_difRcd = 0;
PFDR_difTol = 1e-3*CP_difTol;
PFDR_itMax = 1e4;
PFDR_verbose = 1e2;

%%%  initialize data  %%%
% For details on the data and parameters, see H. Raguet, A Note on the
% Forward-Douglas--Rachford Splitting for Monotone Inclusion and Convex
% Optimization Optimization Letters, 2018, 1-24
load('../data/labeling_3D.mat')

% compute prediction performance of random forest
[~, ML] = max(y, [], 1);
F1 = zeros(1, length(classId));
for k=1:length(classId)
    predk = ML == classId(k);
    truek = ground_truth == classId(k);
    F1(k) = 2*sum(predk & truek)/(sum(predk) + sum(truek));
end
fprintf('\naverage F1 of random forest prediction: %.2f\n\n', mean(F1));
clear predk truek

%%%  solve the optimization problem  %%%
tic;
loss_weights = []; d1_coor_weights = [];
[cv, rx] = cp_pfdr_d1_lsx_mex(loss, y, first_edge, ...
    adj_vertices, homo_d1_weight, loss_weights, d1_coor_weights, ...
    CP_difTol, CP_itMax, PFDR_rho, PFDR_condMin, PFDR_difRcd, ...
    PFDR_difTol, PFDR_itMax, PFDR_verbose);
time = toc;
x = rx(:, cv+1); % rx is components values, cv is components indices
clear cv rx;
fprintf('Total MEX execution time %.0f s\n\n', time);

% compute prediction performance of spatially regularized prediction
[~, ML] = max(x, [], 1);
F1 = zeros(1, length(classId));
for k=1:length(classId)
    predk = ML == classId(k);
    truek = ground_truth == classId(k);
    F1(k) = 2*sum(predk & truek)/(sum(predk) + sum(truek));
end
fprintf('\naverage F1 of spatially regularized prediction: %.2f\n\n', mean(F1));
clear predk truek
