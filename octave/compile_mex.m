origDir = pwd; % remember working directory
cd(fileparts(which('compile_mex.m'))); 
mkdir('bin/');
try
    % compilation flags 
    [~, CXXFLAGS] = system('mkoctfile -p CXXFLAGS');
    [~, LDFLAGS] = system('mkoctfile -p LDFLAGS');
    % some versions introduces a newline character (10)
    % in the output of 'system'; this must be removed
    if CXXFLAGS(end)==10, CXXFLAGS = CXXFLAGS(1:end-1); end
    if LDFLAGS(end)==10, LDFLAGS = LDFLAGS(1:end-1); end
    CXXFLAGSorig = CXXFLAGS;
    LDFLAGSorig = LDFLAGS;
    CXXFLAGS = sprintf('%s %s', CXXFLAGS, '-Wall -fopenmp');
    LDFLAGS = sprintf('%s %s', LDFLAGS, '-fopenmp');
    setenv('CXXFLAGS', CXXFLAGS);
    setenv('LDFLAGS', LDFLAGS);

    %{
    mex mex/cp_pfdr_d1_ql1b_mex.cpp ../src/cp_pfdr_d1_ql1b.cpp ...
        ../src/cut_pursuit_d1.cpp ../src/cut_pursuit.cpp ...
        ../src/cp_graph.cpp ../src/pfdr_d1_ql1b.cpp ../src/matrix_tools.cpp ...
        ../src/pfdr_graph_d1.cpp ../src/pcd_fwd_doug_rach.cpp ...
        ../src/pcd_prox_split.cpp ...
        -output bin/cp_pfdr_d1_ql1b_mex
    clear cp_pfdr_d1_ql1b_mex
    %}

    % %{
    mex mex/cp_pfdr_d1_lsx_mex.cpp ../src/cp_pfdr_d1_lsx.cpp ...
        ../src/cut_pursuit_d1.cpp ../src/cut_pursuit.cpp ...
        ../src/cp_graph.cpp ../src/pfdr_d1_lsx.cpp ../src/proj_simplex.cpp ...
        ../src/pfdr_graph_d1.cpp ../src/pcd_fwd_doug_rach.cpp ...
        ../src/pcd_prox_split.cpp ...
        -output bin/cp_pfdr_d1_lsx_mex
    clear cp_pfdr_d1_lsx_mex
    %}

    %{
    mex mex/cp_kmpp_d0_dist_mex.cpp ../src/cp_kmpp_d0_dist.cpp ...
        ../src/cut_pursuit_d0.cpp ../src/cut_pursuit.cpp ...
        ../src/cp_graph.cpp -output bin/cp_kmpp_d0_dist_mex
    clear cp_kmpp_d0_dist_mex
    %}

    system('rm *.o')
catch % if an error occur, makes sure not to change the working directory
    % back to original environment
    setenv('CXXFLAGS', CXXFLAGSorig);
    setenv('LDFLAGS', LDFLAGSorig);
    cd(origDir);
	rethrow(lasterror);
end
% back to original environment
setenv('CXXFLAGS', CXXFLAGSorig);
setenv('LDFLAGS', LDFLAGSorig);
cd(origDir);
