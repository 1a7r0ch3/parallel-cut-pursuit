#include <cstdlib>
#include <limits>
#include <cstdint> // for instantiation
#include "../include/cp_graph.hpp"

#define TPL template <typename real_t, typename index_t, typename comp_t, \
    typename value_t>
#define CP_GRAPH Cp_graph<real_t, index_t, comp_t, value_t>

/* constants of the correct type */
#define ZERO ((real_t) 0.0)
/* special constants for node->parent */
#define TERMINAL terminal // used to be ((arc *) 1)
#define ORPHAN orphan // used to be ((arc *) 2)
/* infinite distance to the terminal */
#define INFINITE_D (std::numeric_limits<index_t>::max())

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

/***********************************************************************/

/*
	Functions for processing active list.
	i->next points to the next node in the list
	(or to i, if i is the last node in the list).
	If i->next is nullptr iff i is not in the list.

	There are two queues. Active nodes are added
	to the end of the second queue and read from
	the front of the first queue. If the first queue
	is empty, it is replaced by the second queue
	(and the second queue becomes empty).
*/

TPL inline void CP_GRAPH::set_active(node *i)
{
	if (!i->next)
	{
		/* it's not in the list yet */
		if (queue_last[1]) queue_last[1] -> next = i;
		else               queue_first[1]        = i;
		queue_last[1] = i;
		i -> next = i;
	}
}

/*
	Returns the next active node.
	If it is connected to the sink, it stays in the list,
	otherwise it is removed from the list
*/
TPL inline typename CP_GRAPH::node* CP_GRAPH::next_active()
{
	node *i;

	while ( 1 )
	{
		if (!(i=queue_first[0]))
		{
			queue_first[0] = i = queue_first[1];
			queue_last[0]  = queue_last[1];
			queue_first[1] = nullptr;
			queue_last[1]  = nullptr;
			if (!i) return nullptr;
		}

		/* remove it from the active list */
		if (i->next == i) queue_first[0] = queue_last[0] = nullptr;
		else              queue_first[0] = i -> next;
		i -> next = nullptr;

		/* a node in the list is active iff it has a parent */
		if (i->parent) return i;
	}
}

/***********************************************************************/

TPL inline void CP_GRAPH::set_orphan_front(node *i)
{
	nodeptr *np;
	i -> parent = ORPHAN;
	np = nodeptr_block -> New();
	np -> ptr = i;
	np -> next = orphan_first;
	orphan_first = np;
}

TPL inline void CP_GRAPH::set_orphan_rear(node *i)
{
	nodeptr *np;
	i -> parent = ORPHAN;
	np = nodeptr_block -> New();
	np -> ptr = i;
	if (orphan_last) orphan_last -> next = np;
	else             orphan_first        = np;
	orphan_last = np;
	np -> next = nullptr;
}

/***********************************************************************/

TPL void CP_GRAPH::maxflow_init(index_t comp_size, const index_t *comp_nodes)
{
	node *i;
    index_t ii;
    const index_t iimax = comp_size ? comp_size : node_num;

	queue_first[0] = queue_last[0] = nullptr;
	queue_first[1] = queue_last[1] = nullptr;
	orphan_first = nullptr;

	TIME = 0;

	/* for (i=nodes; i<node_last; i++) */
	for (ii=0; ii<iimax; ii++)
	{
        i = (comp_nodes) ? (nodes + comp_nodes[ii]) : (nodes + ii);
		i -> next = nullptr;
		i -> TS = TIME;
		if (i->tr_cap > ZERO)
		{
			/* i is connected to the source */
			i -> is_sink = false;
			i -> parent = TERMINAL;
			set_active(i);
			i -> DIST = 1;
		}
		else if (i->tr_cap < ZERO)
		{
			/* i is connected to the sink */
			i -> is_sink = true;
			i -> parent = TERMINAL;
			set_active(i);
			i -> DIST = 1;
		}
		else
		{
			i -> parent = nullptr;
		}
	}
}

TPL void CP_GRAPH::augment(arc *middle_arc)
{
	node *i;
	arc *a;
	real_t bottleneck;


	/* 1. Finding bottleneck capacity */
	/* 1a - the source tree */
	bottleneck = middle_arc -> r_cap;
	for (i=middle_arc->sister->head; ; i=a->head)
	{
		a = i -> parent;
		if (a == TERMINAL) break;
		if (bottleneck > a->sister->r_cap) bottleneck = a -> sister -> r_cap;
	}
	if (bottleneck > i->tr_cap) bottleneck = i -> tr_cap;
	/* 1b - the sink tree */
	for (i=middle_arc->head; ; i=a->head)
	{
		a = i -> parent;
		if (a == TERMINAL) break;
		if (bottleneck > a->r_cap) bottleneck = a -> r_cap;
	}
	if (bottleneck > - i->tr_cap) bottleneck = - i -> tr_cap;


	/* 2. Augmenting */
	/* 2a - the source tree */
	middle_arc -> sister -> r_cap += bottleneck;
	middle_arc -> r_cap -= bottleneck;
	for (i=middle_arc->sister->head; ; i=a->head)
	{
		a = i -> parent;
		if (a == TERMINAL) break;
		a -> r_cap += bottleneck;
		a -> sister -> r_cap -= bottleneck;
		if (a->sister->r_cap <= ZERO) // negative value flags an active edge
		{
			set_orphan_front(i); // add i to the beginning of the adoption list
		}
	}
	i -> tr_cap -= bottleneck;
	if (!i->tr_cap)
	{
		set_orphan_front(i); // add i to the beginning of the adoption list
	}
	/* 2b - the sink tree */
	for (i=middle_arc->head; ; i=a->head)
	{
		a = i -> parent;
		if (a == TERMINAL) break;
		a -> sister -> r_cap += bottleneck;
		a -> r_cap -= bottleneck;
		if (a->r_cap <= ZERO) // negative value flags an active edge
		{
			set_orphan_front(i); // add i to the beginning of the adoption list
		}
	}
	i -> tr_cap += bottleneck;
	if (!i->tr_cap)
	{
		set_orphan_front(i); // add i to the beginning of the adoption list
	}


	// flow += bottleneck;
}

/***********************************************************************/

TPL void CP_GRAPH::process_source_orphan(node *i)
{
	node *j;
	arc *a0, *a0_min = nullptr, *a;
	index_t d, d_min = INFINITE_D;

	/* trying to find a new parent */
	for (a0=i->first; a0; a0=a0->next)
	if (a0->sister->r_cap > ZERO)
	{
		j = a0 -> head;
		if (!j->is_sink && (a=j->parent))
		{
			/* checking the origin of j */
			d = 0;
			while ( 1 )
			{
				if (j->TS == TIME)
				{
					d += j -> DIST;
					break;
				}
				a = j -> parent;
				d ++;
				if (a==TERMINAL)
				{
					j -> TS = TIME;
					j -> DIST = 1;
					break;
				}
				if (a==ORPHAN) { d = INFINITE_D; break; }
				j = a -> head;
			}
			if (d<INFINITE_D) /* j originates from the source - done */
			{
				if (d<d_min)
				{
					a0_min = a0;
					d_min = d;
				}
				/* set marks along the path */
				for (j=a0->head; j->TS!=TIME; j=j->parent->head)
				{
					j -> TS = TIME;
					j -> DIST = d --;
				}
			}
		}
	}

	if ((i->parent = a0_min))
	{
		i -> TS = TIME;
		i -> DIST = d_min + 1;
	}
	else
	{
		/* process neighbors */
		for (a0=i->first; a0; a0=a0->next)
		{
            /* negative value indicate active edge in the cut-pursuit sense,
             * and thus a0 link to another component */
            if (a0->r_cap < ZERO){ continue; }

			j = a0 -> head;
			if (!j->is_sink && (a=j->parent))
			{
				if (a0->sister->r_cap > ZERO) set_active(j);
				if (a!=TERMINAL && a!=ORPHAN && a->head==i)
				{
					set_orphan_rear(j); // add j to the end of the adoption list
				}
			}
		}
	}
}

TPL void CP_GRAPH::process_sink_orphan(node *i)
{
	node *j;
	arc *a0, *a0_min = nullptr, *a;
	index_t d, d_min = INFINITE_D;

	/* trying to find a new parent */
	for (a0=i->first; a0; a0=a0->next)
	if (a0->r_cap > ZERO)
	{
		j = a0 -> head;
		if (j->is_sink && (a=j->parent))
		{
			/* checking the origin of j */
			d = 0;
			while ( 1 )
			{
				if (j->TS == TIME)
				{
					d += j -> DIST;
					break;
				}
				a = j -> parent;
				d ++;
				if (a==TERMINAL)
				{
					j -> TS = TIME;
					j -> DIST = 1;
					break;
				}
				if (a==ORPHAN) { d = INFINITE_D; break; }
				j = a -> head;
			}
			if (d<INFINITE_D) /* j originates from the sink - done */
			{
				if (d<d_min)
				{
					a0_min = a0;
					d_min = d;
				}
				/* set marks along the path */
				for (j=a0->head; j->TS!=TIME; j=j->parent->head)
				{
					j -> TS = TIME;
					j -> DIST = d --;
				}
			}
		}
	}

	if ((i->parent = a0_min))
	{
		i -> TS = TIME;
		i -> DIST = d_min + 1;
	}
	else
	{
		/* process neighbors */
		for (a0=i->first; a0; a0=a0->next)
		{
            /* negative value indicate active edge in the cut-pursuit sense,
             * and thus a0 link to another component */
            if (a0->r_cap < ZERO){ continue; }

			j = a0 -> head;
			if (j->is_sink && (a=j->parent))
			{
				if (a0->r_cap > ZERO) set_active(j);
				if (a!=TERMINAL && a!=ORPHAN && a->head==i)
				{
					set_orphan_rear(j); // add j to the end of the adoption list
				}
			}
		}
	}
}

/***********************************************************************/

TPL void CP_GRAPH::maxflow(index_t comp_size, const index_t *comp_nodes)
{
	node *i, *j, *current_node = nullptr;
	arc *a;
	nodeptr *np, *np_next;

	if (!nodeptr_block)
	{
		nodeptr_block = new DBlock<nodeptr>(NODEPTR_BLOCK_SIZE);
	}
    
    maxflow_init(comp_size, comp_nodes);

	// main loop
	while ( 1 )
	{
		if ((i=current_node))
		{
			i -> next = nullptr; /* remove active flag */
			if (!i->parent) i = nullptr;
		}
		if (!i)
		{
			if (!(i = next_active())) break;
		}

		/* growth */
		if (!i->is_sink)
		{
			/* grow source tree */
			for (a=i->first; a; a=a->next)
			if (a->r_cap > ZERO)
			{
				j = a -> head;
				if (!j->parent)
				{
					j -> is_sink = false;
					j -> parent = a -> sister;
					j -> TS = i -> TS;
					j -> DIST = i -> DIST + 1;
					set_active(j);
				}
				else if (j->is_sink) break;
				else if (j->TS <= i->TS &&
				         j->DIST > i->DIST)
				{
					/* heuristic - trying to make the distance from j to the source shorter */
					j -> parent = a -> sister;
					j -> TS = i -> TS;
					j -> DIST = i -> DIST + 1;
				}
			}
		}
		else
		{
			/* grow sink tree */
			for (a=i->first; a; a=a->next)
			if (a->sister->r_cap > ZERO)
			{
				j = a -> head;
				if (!j->parent)
				{
					j -> is_sink = true;
					j -> parent = a -> sister;
					j -> TS = i -> TS;
					j -> DIST = i -> DIST + 1;
					set_active(j);
				}
				else if (!j->is_sink) { a = a -> sister; break; }
				else if (j->TS <= i->TS &&
				         j->DIST > i->DIST)
				{
					/* heuristic - trying to make the distance from j to the sink shorter */
					j -> parent = a -> sister;
					j -> TS = i -> TS;
					j -> DIST = i -> DIST + 1;
				}
			}
		}

        /* changed type from long to index_t */
        /* can't we prove this won't overflow? */
		if (++TIME <= 0){
            std::cerr << "Boykov & Kolmogorov maxflow: timestamp overflow." << std::endl;
            exit(EXIT_FAILURE);
        }

		if (a)
		{
			i -> next = i; /* set active flag */
			current_node = i;

			augment(a);

			/* adoption */
			while ((np=orphan_first))
			{
				np_next = np -> next;
				np -> next = nullptr;

				while ((np=orphan_first))
				{
					orphan_first = np -> next;
					i = np -> ptr;
					nodeptr_block -> Delete(np);
					if (!orphan_first) orphan_last = nullptr;
					if (i->is_sink) process_sink_orphan(i);
					else            process_source_orphan(i);
				}

				orphan_first = np_next;
			}
			/* adoption end */
		}
		else current_node = nullptr;
	}
    
	{
		delete nodeptr_block; 
		nodeptr_block = nullptr; 
	}

	// return flow;
}

/***********************************************************************/


/* instantiate for compilation */
template class Cp_graph<float, uint32_t, uint16_t>;
template class Cp_graph<double, uint32_t, uint16_t>;
template class Cp_graph<float, uint32_t, uint32_t>;
template class Cp_graph<double, uint32_t, uint32_t>;
