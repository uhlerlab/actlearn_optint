import numpy as np
import graphical_models as gm
import networkx as nx
from itertools import combinations


# return topologically orderd DAG
def ordered_DAG(dag):
	nnodes = dag.nnodes

	order = dag.topological_sort()
	ord_dag = gm.DAG(set(range(nnodes)))
	ord_dag.add_arcs_from((order.index(i), order.index(j)) for (i,j) in dag.arcs)

	return ord_dag 


# generate DAGs of different types
def random_graph(nnodes, ratio=.2):
	return gm.rand.directed_erdos(nnodes, ratio) 


def barabasialbert_graph(nnodes, m=2):
	return gm.rand.directed_barabasi(nnodes, m)
	

def line_graph(nnodes):
	dag = gm.DAG(set(range(nnodes)))
	
	perm = np.random.permutation(nnodes)
	orient = np.random.randint(0, 2, nnodes-1)
	dag.add_arcs_from(((perm[i], perm[i+1]), (perm[i+1], perm[i]))[orient[i]] for i in range(nnodes-1))
	
	return dag


def path_graph(nnodes):
	dag = gm.DAG(set(range(nnodes)))
	
	perm = np.random.permutation(nnodes)
	dag.add_arcs_from((perm[i], perm[i+1]) for i in range(nnodes-1))
	
	return dag


def instar_graph(nnodes):
	dag = gm.DAG(set(range(nnodes)))

	perm = np.random.permutation(nnodes)
	dag.add_arcs_from((perm[i+1], perm[0]) for i in range(nnodes-1))	

	return dag


def outstar_graph(nnodes):
	dag = gm.DAG(set(range(nnodes)))

	perm = np.random.permutation(nnodes)
	dag.add_arcs_from((perm[0], perm[i+1]) for i in range(nnodes-1))

	return dag


def tree_graph(nnodes):
	dag = gm.DAG(set(range(nnodes)))

	tree = nx.random_tree(nnodes)
	queue = [0]
	while queue:
		current_node = queue.pop()
		nbrs = list(tree.neighbors(current_node))
		nbr_size = len(nbrs)
		orient = np.random.randint(0, 2, nbr_size)
		dag.add_arcs_from(((current_node, nbrs[i]), (nbrs[i], current_node))[orient[i]] for i in range(nbr_size))
		queue += nbrs
		tree.remove_node(current_node)

	return dag


def complete_graph(nnodes):
	dag = gm.DAG(set(range(nnodes)))

	perm = np.random.permutation(nnodes)
	for (i,j) in combinations(perm, 2):
		dag.add_arc(i,j)
	
	return dag


def chordal_graph(nnodes, density=.2):
	while True:
		d = nx.DiGraph()
		d.add_nodes_from(set(range(nnodes)))
		order = list(range(1, nnodes))
		for i in order:
			num_parents_i = max(1, np.random.binomial(i, density))
			parents_i = np.random.choice(range(i), num_parents_i, replace=False)
			d.add_edges_from({(p, i) for p in parents_i})
		for i in reversed(order):
			for j, k in combinations(d.predecessors(i), 2):
				d.add_edge(min(j, k), max(j, k))

		perm = np.random.permutation(list(range(nnodes)))
		d = nx.relabel.relabel_nodes(d, dict(enumerate(perm)))

		return gm.DAG.from_nx(d)


def rooted_tree_graph(nnodes):
	g = nx.random_tree(nnodes)
	root = np.random.randint(0, nnodes)
	d = nx.DiGraph()

	queue = [root]
	while queue:
		current_node = queue.pop()
		nbrs = list(g.neighbors(current_node))
		d.add_edges_from([(current_node, nbr) for nbr in nbrs])
		queue += nbrs
		g.remove_node(current_node)
	return gm.DAG.from_nx(d)


def cliquetree_graph(nnodes, degree=3, min_clique_size=3, max_clique_size=5):
	counter = np.random.randint(min_clique_size, max_clique_size)
	source_clique = list(range(counter))
	previous_layer_cliques = [source_clique]
	current_layer_cliques = []
	arcs = set(combinations(source_clique, 2))

	while counter < nnodes:
		for parent_clique in previous_layer_cliques:
			for d in range(degree):
				if counter < nnodes:
					clique_size = np.random.randint(min_clique_size, max_clique_size)
					intersection_size = min(len(parent_clique)-1, np.random.randint(int(clique_size/2), clique_size-1))
					num_new_nodes = clique_size - intersection_size

					indices = set(np.random.choice(parent_clique, intersection_size, replace=False))
					intersection = [
						parent_clique[ix] for ix in range(len(parent_clique))
						if ix in indices
					]
					new_clique = intersection + list(range(counter, counter+num_new_nodes))
					current_layer_cliques.append(new_clique)
					arcs.update(set(combinations(new_clique, 2)))
					counter += num_new_nodes
		previous_layer_cliques = current_layer_cliques.copy()
	g = nx.DiGraph()
	g.add_edges_from(arcs)
	# if not nx.is_connected(g.to_undirected()):
		# raise RuntimeError
	return gm.DAG.from_nx(g)

