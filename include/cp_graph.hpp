/*==========================================================================
 * Boykov & Kolmogorov graph structure and maximum flow algorithm.
 * Modified by Hugo Raguet for use with cut-pursuit algorithms (2016, 2018)
 *
 * - reuse tree option removed, together with dedicated variables
 * - nodes and arcs arrays made public for direct manipulation by cut-pursuit
 * - cannot reallocate nodes and arcs array for now
 * - max flows are not computed anymore, since they are not useful here
 * - capacities are all of type real_t, nodes and edges are indexed by
 *   integral type index_t, and a type comp_t is used to index components
 * - DIST can be accessed as 'comp', for storing component assignment
 * - TS can be accessed as 'vertex', for storing components lists
 * - is_sink can be accessed as 'saturation', for flagged saturated components
 * - arc residual capacity can be assigned to negative value for flagging
 *   "active" edges (in the cut-pursuit sense); note that with such a negative
 *   capacity, no augmenting path can go through activated edges, and thus
 *   the capacity will stay negative when maxflow() is computed; in
 *   particular, max flows can be computed independently over each cut-pursuit 
 *   component; beware however that this must be enforced when processing
 *   orphans (see process_*_orphan() in cp_maxflow.cpp)
 * - a derived class Cp_graph_parallel is implemented, useful to handle
 *   private copies of a main graph in parallel threads
 *
 * some other modifications:
 *  - do not initialize nodes array with memset() because the null pointer
 *    is not guaranteed all-bits-zero ; explicit initialization instead
 *  - create explicit TERMINAL and ORPHAN sentinel pointers because there is no
 *    absolute guarantee that adresses ((arc*) 1) or ((arc*) 2) are not used
 *
 * ========================

   Copyright Vladimir Kolmogorov (vnk@ist.ac.at), Yuri Boykov (yuri@csd.uwo.ca)

   This file is modified from MAXFLOW.

   MAXFLOW is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   MAXFLOW is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with MAXFLOW.  If not, see <http://www.gnu.org/licenses/>.       
*============================================================================*/
#pragma once
#include <iostream>
#include "block.hpp"

/* declare cut-pursuit base class for friendship */
template <typename real_t, typename index_t, typename comp_t> class Cp;

/* real_t is the real numeric type, used for objective functional computation
 * and thus for edge weights and flow graph capacities;
 * index_t must be able to represent the number of vertices and of (undirected)
 * edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph */
template <typename real_t, typename index_t, typename comp_t> class Cp_graph
{
    friend class Cp<real_t, index_t, comp_t>;

public:

    /* used to be a long int; added an overflow check in cp_maxflow.cpp */
    typedef index_t timestamp_t; 

	Cp_graph(index_t node_num_max, index_t edge_num_max); // constructor

    /* copy constructor allows instantiation of private subgraphs pointing to
     * an already existing graph for calling maxflow() in parallel threads */
    Cp_graph(const Cp_graph & G);

    /* destructor; warning: if it is a parallel copy, it does not delete the
     * nodes and arcs arrays; however if it is an original graph, it does not
     * check if parallel copies still exists in memory */
	~Cp_graph();

	/* Adds node(s) to the graph. By default, one node is added (num=1);
     * then first call returns 0, second call returns 1, and so on. 
	 * If num > 1, then several nodes are added, and index_t of the first one
	 * is returned. */
	index_t add_node(index_t num = 1);

	/* Adds a bidirectional edge between 'i' and 'j' with the weights 'cap'
     * and 'rev_cap' */
	void add_edge(index_t i, index_t j, real_t cap, real_t rev_cap);

	// Adds new edges 'SOURCE->i' and 'i->SINK' with corresponding weights.
	// Can be called multiple times for each node.
	// Weights can be negative.
	// NOTE: the number of such edges is not counted in edge_num_max.
	void add_tweights(index_t i, real_t cap_source, real_t cap_sink);

	// Computes the maxflow. Can be called several times.
	void maxflow(index_t comp_size = 0, const index_t *comp_nodes = nullptr);

private:

    struct node;
    struct arc;

	struct node
	{
		arc* first; // first outcoming arc
		arc* parent; // node's parent
		node* next; // pointer to the next active node
                          // (or to itself if it is the last node in the list)
        union {
            timestamp_t TS; // timestamp showing when DIST was computed
            index_t vertex; // temporarily store components list
                            // (useful for computing connected components)
        };
        union {
            index_t DIST; // distance to the terminal
            comp_t comp; // temporarily store components assignment
                         // (useful for computing iterate evolution)
        };
        union {
            bool is_sink; // flag showing whether the node is in the source
                          // or in the sink tree (if parent is not null)
            bool saturation; // saturated components (which cannot be cut) are
                             // flagged on their first vertex
        };
        /* if tr_cap > 0, tr_cap is residual capacity of the arc SOURCE->node
         * otherwise     -tr_cap is residual capacity of the arc node->SINK */
		real_t tr_cap;
	};

	struct arc
	{
		node* head; // node the arc points to
		arc* next; // next arc with the same originating node
		arc* sister; // reverse arc
		real_t r_cap; // residual capacity
	};

	node* nodes;
	arc* arcs;

    /* special constants for node->parent */
    arc reserved_terminal_arc;
    arc reserved_orphan_arc;
    arc* const terminal;
    arc* const orphan;

    /* node_last = nodes+node_num, node_max = nodes+node_num_max */
	node *node_last, *node_max;
    /* arc_last = arcs+2*edge_num, arc_max = arcs+2*edge_num_max */
	arc *arc_last, *arc_max;

	index_t node_num = 0;

	struct nodeptr
	{
		node* ptr;
		nodeptr* next;
	};
	static const int NODEPTR_BLOCK_SIZE = 128;


	DBlock<nodeptr> *nodeptr_block;

	// real_t flow; // total flow; not used for cut-pursuit

	node *queue_first[2], *queue_last[2]; // list of active nodes
	nodeptr *orphan_first, *orphan_last; // list of pointers to orphans
	timestamp_t TIME;	// monotonically increasing global counter

	// functions for processing active list
	void set_active(node *i);
	node *next_active();

	// functions for processing orphans list
	void set_orphan_front(node* i); // add to the beginning of the list
	void set_orphan_rear(node* i);  // add to the end of the list

	void maxflow_init(index_t comp_size = 0,
        const index_t *comp_nodes = nullptr);
	void augment(arc *middle_arc);
	void process_source_orphan(node *i);
	void process_sink_orphan(node *i);

    bool is_parallel_copy; // flag a private copy in a parallel thread
};

#define TPL template <typename real_t, typename index_t, typename comp_t>
#define CP_GRAPH Cp_graph<real_t, index_t, comp_t>

/***  inline methods  ***/

TPL inline index_t CP_GRAPH::add_node(index_t num)
{
	if (node_last + num > node_max){
        std::cerr << "Boykov & Kolmogorov graph: "
            << node_last - nodes + num
            << " nodes allocated, but "
            << node_max - nodes << " nodes requested." << std::endl;
        exit(EXIT_FAILURE);
    }

    for (index_t n = 0; n < num; n++){ node_last[n].first = nullptr; }

	index_t i = node_num;
	node_num += num;
	node_last += num;
    
	return i;
}

TPL inline void CP_GRAPH::add_tweights(index_t i, real_t cap_source,
    real_t cap_sink)
{
	real_t delta = nodes[i].tr_cap;
	if (delta > 0) cap_source += delta;
	else           cap_sink   -= delta;
	// flow += (cap_source < cap_sink) ? cap_source : cap_sink;
	nodes[i].tr_cap = cap_source - cap_sink;
}

TPL inline void CP_GRAPH:: add_edge(index_t _i, index_t _j, real_t cap,
    real_t rev_cap)
{
	if (arc_last == arc_max){
        std::cerr << "Boykov & Kolmogorov graph: all " << arc_max - arcs
            << " allocated arcs are assigned, but more arcs are requested."
            << std::endl;
        exit(EXIT_FAILURE);
    }

	arc *a = arc_last ++;
	arc *a_rev = arc_last ++;

	node* i = nodes + _i;
	node* j = nodes + _j;

	a -> sister = a_rev;
	a_rev -> sister = a;
	a -> next = i -> first;
	i -> first = a;
	a_rev -> next = j -> first;
	j -> first = a_rev;
	a -> head = j;
	a_rev -> head = i;
	a -> r_cap = cap;
	a_rev -> r_cap = rev_cap;
}

#undef TPL
#undef CP_GRAPH
