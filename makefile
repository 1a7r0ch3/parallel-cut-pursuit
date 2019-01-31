CC=g++
#CFLAGS=-Wl,--no-as-needed -std=c++11 -g -O2 -fopenmp
CFLAGS=-std=c++11 -g -O2 -fopenmp
LIBS=-lpthread
SRC=$(PWD)/src
INCLUDE=$(PWD)/include
MAIN=$(PWD)/example
BUILDDIR=$(PWD)/build
OUTDIR=$(BUILDDIR)/bin
TEMPDIR=$(BUILDDIR)/tmp
MKDIR_P = mkdir -p

all: directories $(OUTDIR)/test

directories: ${OUTDIR} $(TEMPDIR)
$(OUTDIR): 
	    ${MKDIR_P} $(OUTDIR)
$(TEMPDIR):
	    ${MKDIR_P} $(TEMPDIR)

$(OUTDIR)/test: $(TEMPDIR)/test.o $(TEMPDIR)/cp_pfdr_d1_ql1b.o $(TEMPDIR)/matrix_tools.o $(TEMPDIR)/cut_pursuit_d1.o $(TEMPDIR)/cut_pursuit.o $(TEMPDIR)/cp_graph.o $(TEMPDIR)/cp_maxflow.o $(TEMPDIR)/pfdr_d1_ql1b.o $(TEMPDIR)/pfdr_graph_d1.o $(TEMPDIR)/pcd_fwd_doug_rach.o $(TEMPDIR)/pcd_prox_split.o
	$(CC) $(CFLAGS) $(TEMPDIR)/test.o $(TEMPDIR)/cp_pfdr_d1_ql1b.o $(TEMPDIR)/matrix_tools.o $(TEMPDIR)/cut_pursuit_d1.o $(TEMPDIR)/cut_pursuit.o $(TEMPDIR)/cp_graph.o $(TEMPDIR)/cp_maxflow.o $(TEMPDIR)/pfdr_d1_ql1b.o $(TEMPDIR)/pfdr_graph_d1.o $(TEMPDIR)/pcd_fwd_doug_rach.o $(TEMPDIR)/pcd_prox_split.o $(LIBS) -o $(OUTDIR)/test

$(TEMPDIR)/test.o: $(MAIN)/test.cpp $(INCLUDE)/cp_pfdr_d1_ql1b.hpp $(INCLUDE)/omp_num_threads.hpp $(INCLUDE)/matrix_tools.hpp $(INCLUDE)/wth_element.hpp $(SRC)/wth_element_generic.cpp $(INCLUDE)/cut_pursuit_d1.hpp $(INCLUDE)/cut_pursuit.hpp $(INCLUDE)/cp_graph.hpp $(INCLUDE)/block.hpp 
	$(CC) -c $(CFLAGS) $(MAIN)/test.cpp -o $(TEMPDIR)/test.o

$(TEMPDIR)/cp_pfdr_d1_ql1b.o: $(SRC)/cp_pfdr_d1_ql1b.cpp $(INCLUDE)/cp_pfdr_d1_ql1b.hpp $(INCLUDE)/omp_num_threads.hpp $(INCLUDE)/matrix_tools.hpp $(INCLUDE)/wth_element.hpp $(SRC)/wth_element_generic.cpp $(INCLUDE)/cut_pursuit_d1.hpp $(INCLUDE)/cut_pursuit.hpp $(INCLUDE)/cp_graph.hpp $(INCLUDE)/block.hpp 
	$(CC) -c $(CFLAGS) $(SRC)/cp_pfdr_d1_ql1b.cpp -o $(TEMPDIR)/cp_pfdr_d1_ql1b.o

$(TEMPDIR)/matrix_tools.o: $(SRC)/matrix_tools.cpp $(INCLUDE)/matrix_tools.hpp $(INCLUDE)/omp_num_threads.hpp 
	$(CC) -c $(CFLAGS) $(SRC)/matrix_tools.cpp -o $(TEMPDIR)/matrix_tools.o

$(TEMPDIR)/cut_pursuit_d1.o: $(SRC)/cut_pursuit_d1.cpp $(INCLUDE)/cut_pursuit_d1.hpp $(INCLUDE)/omp_num_threads.hpp 
	$(CC) -c $(CFLAGS) $(SRC)/cut_pursuit_d1.cpp -o $(TEMPDIR)/cut_pursuit_d1.o

$(TEMPDIR)/cut_pursuit.o: $(SRC)/cut_pursuit.cpp $(INCLUDE)/cut_pursuit.hpp $(INCLUDE)/omp_num_threads.hpp 
	$(CC) -c $(CFLAGS) $(SRC)/cut_pursuit.cpp -o $(TEMPDIR)/cut_pursuit.o

$(TEMPDIR)/cp_graph.o: $(SRC)/cp_graph.cpp $(INCLUDE)/cp_graph.hpp $(INCLUDE)/block.hpp 
	$(CC) -c $(CFLAGS) $(SRC)/cp_graph.cpp -o $(TEMPDIR)/cp_graph.o

$(TEMPDIR)/cp_maxflow.o: $(SRC)/cp_maxflow.cpp $(INCLUDE)/cp_graph.hpp $(INCLUDE)/block.hpp 
	$(CC) -c $(CFLAGS) $(SRC)/cp_maxflow.cpp -o $(TEMPDIR)/cp_maxflow.o

$(TEMPDIR)/pfdr_d1_ql1b.o: $(SRC)/pfdr_d1_ql1b.cpp $(INCLUDE)/pfdr_d1_ql1b.hpp $(INCLUDE)/matrix_tools.hpp  $(INCLUDE)/omp_num_threads.hpp $(INCLUDE)/pfdr_graph_d1.hpp $(INCLUDE)/pcd_fwd_doug_rach.hpp $(INCLUDE)/pcd_prox_split.hpp 
	$(CC) -c $(CFLAGS) $(SRC)/pfdr_d1_ql1b.cpp -o $(TEMPDIR)/pfdr_d1_ql1b.o

$(TEMPDIR)/pfdr_graph_d1.o: $(SRC)/pfdr_graph_d1.cpp $(INCLUDE)/omp_num_threads.hpp $(INCLUDE)/pfdr_graph_d1.hpp $(INCLUDE)/pcd_fwd_doug_rach.hpp $(INCLUDE)/pcd_prox_split.hpp
	$(CC) -c $(CFLAGS) $(SRC)/pfdr_graph_d1.cpp -o $(TEMPDIR)/pfdr_graph_d1.o

$(TEMPDIR)/pcd_fwd_doug_rach.o: $(SRC)/pcd_fwd_doug_rach.cpp $(INCLUDE)/omp_num_threads.hpp $(INCLUDE)/pcd_fwd_doug_rach.hpp $(INCLUDE)/pcd_prox_split.hpp
	$(CC) -c $(CFLAGS) $(SRC)/pcd_fwd_doug_rach.cpp -o $(TEMPDIR)/pcd_fwd_doug_rach.o

$(TEMPDIR)/pcd_prox_split.o: $(SRC)/pcd_prox_split.cpp $(INCLUDE)/pcd_prox_split.hpp $(INCLUDE)/omp_num_threads.hpp 
	$(CC) -c $(CFLAGS) $(SRC)/pcd_prox_split.cpp -o $(TEMPDIR)/pcd_prox_split.o

clean:
	    rm -rf $(BUILDDIR)

