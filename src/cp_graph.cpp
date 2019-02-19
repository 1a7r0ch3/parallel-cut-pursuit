#include <cstdlib>
#include <cstdint> // for instantiation
#include <iostream>
#include "../include/cp_graph.hpp"

#define TPL template <typename real_t, typename index_t, typename comp_t>
#define CP_GRAPH Cp_graph<real_t, index_t, comp_t>

using namespace std;

TPL CP_GRAPH::Cp_graph(index_t node_num_max, index_t edge_num_max)
    : node_num(0), nodeptr_block(nullptr), terminal(&reserved_terminal_arc),
    orphan(&reserved_orphan_arc), is_parallel_copy(false)
{
    if (node_num_max < 16) node_num_max = 16;
    if (edge_num_max < 16) edge_num_max = 16;

    nodes = (node*) malloc(sizeof(node)*node_num_max);
    arcs = (arc*) malloc(sizeof(arc)*2*edge_num_max);
    if (!nodes || !arcs) {
        cerr << "Boykov & Kolmogorov graph: not enough memory." << endl;
        exit(EXIT_FAILURE);
    }

	node_last = nodes;
	node_max = nodes + node_num_max;
	arc_last = arcs;
	arc_max = arcs + 2*edge_num_max;

    is_parallel_copy = false;

	// flow = 0;
}

TPL CP_GRAPH::Cp_graph(const CP_GRAPH & G) : nodes(G.nodes), arcs(G.arcs),
    node_num(G.node_num), terminal(G.terminal), orphan(G.orphan),
    nodeptr_block(nullptr), is_parallel_copy(true){}

TPL CP_GRAPH::~Cp_graph()
{
	if (nodeptr_block) 
	{ 
		delete nodeptr_block; 
		nodeptr_block = nullptr; 
	}
    if (!is_parallel_copy){
        free(nodes);
        free(arcs);
    }
}

/* instantiate for compilation */
template class Cp_graph<float, uint32_t, uint16_t>;

template class Cp_graph<double, uint32_t, uint16_t>;

template class Cp_graph<float, uint32_t, uint32_t>;

template class Cp_graph<double, uint32_t, uint32_t>;
