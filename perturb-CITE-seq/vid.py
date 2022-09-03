# -*- coding: utf-8 -*-
"""
@author: louis.cammarata
"""

from turtle import color
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

def draw(pdag, node_names=None):
    """ Function to plot a pdag
    
    Args:
        pdag: the pdag or dag to be plotted
        
    Returns:
        Plot of the pdag
    
    """
    
    p = pdag.nnodes
        
    nw_ax = plt.subplot2grid((10, 10), (0, 0), colspan=9, rowspan=9)
    
    plt.gcf().set_size_inches(20, 5)
    

    d = nx.DiGraph()
    for i in pdag.nodes:
        d.add_node(i)
    for (i, j) in pdag.arcs:
        d.add_edge(i, j)

    e = nx.Graph()
    for pair in pdag.edges:
        (i, j) = tuple(pair)
        e.add_edge(i, j)
    
    pos = graphviz_layout(d, 'dot')
    nx.draw(e, pos=pos, node_color='w', ax=nw_ax, edge_color='tab:red')
    nx.draw(d, pos=pos, node_color='tab:blue', alpha=0.5, ax=nw_ax)
    if node_names is not None:
        labels={node: node_names[node] for node in range(p)}
    else:
        labels={node:node for node in pdag.nodes}
    nx.draw_networkx_labels(d, pos, labels=labels, ax=nw_ax, font_size=10, font_color="black")