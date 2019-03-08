         %----------------------------------------------------------%
         %  script for illustrating cp_pfdr_d1_ql1b on EEG problem  %
         %----------------------------------------------------------%
% Reference: H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
% Nonsmooth Functionals with Graph Total Variation, International Conference on
% Machine Learning, PMLR, 2018, 80, 4244-4253
%
% Hugo Raguet 2017, 2018
cd(fileparts(which('example_EEG.m')));
addpath(genpath('./bin/'));

%%%  general parameters  %%%
printResults = false; % requires color encapsulated postscript driver on your
                      % system; be sure to run octave 4.2.2 or later, fixing a
                      % bug in trisurf

% parameters for colormap
numberOfColors = 256;
darkLevel = 1/16;

%%%  parameters; see octave/doc/CP_PFDR_graph_d1_l1  %%%
CP_difTol = 1e-4;
CP_itMax = 15;
PFDR_rho = 1.5;
PFDR_condMin = 1e-2;
PFDR_difRcd = 0;
PFDR_difTol = 1e-3*CP_difTol;
PFDR_itMax = 1e4;
PFDR_verbose = 1e3;

%%%  initialize data  %%%
% dataset courtesy of Ahmad Karfoul and Isabelle Merlet, LTSI, INSERM U1099
% Penalization parameters computed with SURE methods, heuristics adapted from
% H. Raguet: A Signal Processing Approach to Voltage-Sensitive Dye Optical
% Imaging, Ph.D. Thesis, Paris-Dauphine University, 2014
load('../data/EEG.mat')

supp0 = x0 ~= 0; % ground truth support 

% the following creates a colormap adapted to representation of sparse data
n = numberOfColors/3;
% luminance of pure red is 0.2989
redMap = [linspace(darkLevel/0.2989, 1, floor(n))'; ones(round(n) + ceil(n), 1)];
greenMap = [zeros(floor(n), 1);  (1:round(n))'/round(n); ones(ceil(n), 1)];
blueMap = [zeros(floor(n) + round(n), 1); (1:ceil(n))'/ceil(n)];
colMap = [redMap, greenMap, blueMap];
colMap = [darkLevel*[1, 1, 1]; colMap]; % this is luminance of pure red [1 0 0];
clear redMap greenMap blueMap
% plot the ground truth and estimated brain sources
CAM = 1.0e+03*[-0.6329    1.5675    0.1686]; % camera parameter well adapted to
                                             % the sources distribution
% get absolute min and max values to plot with same colormap
x0min = min(x0); 
x0max = max(x0);

% print the ground truth activity
figure(1), clf, colormap(colMap);
% map the color index
xcol = floor((x0 - x0min)/(x0max - x0min)*numberOfColors) + 2;
xcol(~supp0) = 1;
% require octave 4.2.2 or later, fixing a bug in trisurf
trisurf(mesh.f, mesh.v(:,1), mesh.v(:,2), mesh.v(:,3), xcol, 'CDataMapping', 'direct');
set(gca, 'Color', 'none'); axis off;
set(gca, 'CameraPosition', CAM);
drawnow('expose');
if printResults
    fprintf('print ground truth... ')
    print(gcf, '-depsc', 'EEG_ground_truth');
    fprintf('done.\n');
end

%%%  solve the optimization problem  %%%
tic;
Yl1 = []; Lbnd = 0.0; Ubnd = Inf;
[cv, rx] = cp_pfdr_d1_ql1b_mex(y, Phi, first_edge, ...
    adj_vertices, d1_weights, Yl1, l1_weights, Lbnd, Ubnd, CP_difTol, ...
    CP_itMax, PFDR_rho, PFDR_condMin, PFDR_difRcd, PFDR_difTol, PFDR_itMax, ...
    PFDR_verbose);
time = toc;
x = rx(cv+1); % rx is components values, cv is components indices
clear cv rx;
fprintf('Total MEX execution time %.1f s\n\n', time);

%%%  compute Dice scores  %%%
% support retrieved with raw model
supp = x ~= 0;
DS = 2*sum(supp0 & supp)/(sum(supp0) + sum(supp));
% support retrieved by discarding nonsignificant values with 2-means clustering
abss = abs(x);
sabs = sort(abss);
n0 = 0; n1 = length(x0); % number of elements per cluster
sum0 = 0; sum1 = sum(sabs); % sum of each cluster
m = sum1/n1;
while 2*sabs(n0+1) < m
    n0 = n0 + 1;
    n1 = n1 - 1;
    sum0 = sum0 + sabs(n0);
    sum1 = sum1 - sabs(n0);
    m = (sum0/n0 + sum1/n1);
end
suppa = abss > (m/2);
DSa = 2*sum(supp0 & suppa)/(sum(supp0) + sum(suppa));
fprintf('Dice score: raw %.2f; approx (discard less significant with 2-means) %.2f\n\n', DS, DSa);

% print retrieved activity
figure(2), clf, colormap(colMap);
xcol = floor((x - x0min)/(x0max - x0min)*numberOfColors) + 2;
xcol(~supp) = 1;
% be sure to run octave 4.2.2 or later, fixing a bug in trisurf
trisurf(mesh.f, mesh.v(:,1), mesh.v(:,2), mesh.v(:,3), xcol, 'CDataMapping', 'direct');
set(gca, 'Color', 'none'); axis off;
set(gca, 'CameraPosition', CAM);
drawnow('expose');
if printResults
    fprintf('print retrieved brain activity... ');
    print(gcf, '-depsc', 'EEG_brain_activity');
    fprintf('done.\n');
end

% print retrieved support
figure(3), clf, colormap(colMap);
xcol = 1 + suppa*numberOfColors;
% be sure to run octave 4.2.2 or later, fixing a bug in trisurf
trisurf(mesh.f, mesh.v(:,1), mesh.v(:,2), mesh.v(:,3), xcol, 'CDataMapping', 'direct');
set(gca, 'Color', 'none'); axis off;
set(gca, 'CameraPosition', CAM);
drawnow('expose');
if printResults
    fprintf('print retrieved brain sources... ')
    print(gcf, '-depsc', 'EEG_brain_sources');
    fprintf('done.\n');
end
