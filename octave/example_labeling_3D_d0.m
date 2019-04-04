  %-------------------------------------------------------------------------%
  %  script for illustrating cp_kmpp_d0_dist on labeling of 3D point cloud  %
  %-------------------------------------------------------------------------%
% Reference: TODO
%
% Hugo Raguet 2019
cd(fileparts(which('example_labeling_3D.m')));
addpath(genpath('./bin/'));

%%%  classes involved in the task  %%%
classNames = {'road', 'vegetation', 'facade', 'hardscape', 'scanning artifacts', 'cars'};
classId = uint8(1:6)';

%%%  parameters; see octave/doc/cp_pfdr_d1_lsx_mex.m  %%%
cp_dif_tol = 1e-3;
cp_it_max = 10;
K = 2;
split_iter_num = 2;
kmpp_init_num = 3;
kmpp_iter_num = 3;
verbose = 1;

%%%  initialize data  %%%
% For details on the data and parameters, see H. Raguet, A Note on the
% Forward-Douglas--Rachford Splitting for Monotone Inclusion and Convex
% Optimization Optimization Letters, 2018, 1-24
load('../data/labeling_3D.mat')
homo_d0_weight = 3*homo_d1_weight; % adjusted for d1 norm by trial-and-error

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
vert_weights = []; coor_weights = [];
[cv, rx] = cp_kmpp_d0_dist_mex(loss, y, first_edge, adj_vertices, ...
    homo_d0_weight, vert_weights, coor_weights, cp_dif_tol, cp_it_max, ...
    K, split_iter_num, kmpp_init_num, kmpp_iter_num, verbose);
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
