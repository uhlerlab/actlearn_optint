import networkx as nx
import graphical_models as gm
import pickle
from copy import deepcopy


PATHDAG = "../../data/estimated_dag.pkl"

with open(PATHDAG, 'rb') as file:
    dag = pickle.load(file)
    
# fully connected graph 
fully_connected_dag = deepcopy(dag)

topological_order = list(nx.topological_sort(dag))
for i in range(len(topological_order)):
    for j in range(i+1, len(topological_order), 1):
        node_i = topological_order[i]
        node_j = topological_order[j]
        if not fully_connected_dag.has_edge(node_i, node_j):
            fully_connected_dag.add_edge(node_i, node_j)

PATHDAG = "../../data/estimated_fully_connected_dag.pkl"

with open(PATHDAG, 'wb') as file:
    pickle.dump(fully_connected_dag, file)