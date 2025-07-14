import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score

def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def binary_auROC(clf, input_data, binary_target):
    return roc_auc_score(binary_target, clf.predict_proba(input_data)[:,1])


def dot_bracket_to_graph(dot_bracket):
    stack = []
    graph = nx.Graph()
    
    # Add nodes (nucleotides)
    num_nts = len(dot_bracket)
    graph.add_nodes_from(range(num_nts))
    
    # Add edges based on the dot-bracket notation
    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)  # Push the index of '(' onto the stack
        elif char == ')':
            j = stack.pop()  # Pop the matching '(' from the stack
            graph.add_edge(i, j)  # Add edge between paired nucleotides
        
        # Add edges between consecutive nucleotides (adjacency)
        if i > 0:
            graph.add_edge(i, i - 1)  # Add edge between consecutive nucleotides
        
    return graph

def draw_rna_graph(graph):
    pos = nx.kamada_kawai_layout(graph)
    nx.draw(
        graph, pos, with_labels=True, node_color='lightblue',
        node_size=500, font_size=10, edge_color='gray'
    )

def binary_auROC(clf, input_data, binary_target):
    return roc_auc_score(binary_target, clf.predict_proba(input_data)[:,1])