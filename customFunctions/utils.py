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


def feature_name_conversion(feature_name, replace_underscore=False):
    try:
        feature_name = {
            "PhastCons_100V":"PhastCons_score",
            "PhyloP_100V":"PhyloP_score",
            "MLC_score":"MLC_score",
            "ss_distorsion":"Secondary_structure_alteration",
            "ddG":"ΔΔG",
            "ref_frequency":"Reference_Nucleotide_Frequency",
            "alt_frequency":"Alternate_Nucleotide_Frequency",
            "coverage":"Coverage_within_MSA",
            "entropy":"Sequence_Entropy",
            "conservation_blocks":"Phylogenetic_Conservation", #depth?
            "PWM_score":"Position_Weight_Matrix",
            "Human_MSA_position":"Position_in_the_human_MSA",
            "bulge":"Bulge_Structural_Motif",
            "stem":"Stem_Structural_Motif",
            "hairpin":"Hairpin_Structural_Motif",
            "loop":"Loop_Structural_Motif",
            "unpaired_frequency":"Unpaired_Nucleotide_Frequency",
            "closest_modification":"distance_to_PTM_site",
            "closest_modification_2D":"Spatial_distance_to_PTM_site",
            "OEUF_score":"OEUF_score",
            "RNA_MSM":"RNA-MSM",
            "x_coord":"PDB_coordinate_on_x_axis",
            "y_coord":"PDB_coordinate_on_y_axis",
            "z_coord":"PDB_coordinate_on_z_axis"
        }[feature_name]
    except KeyError:
        pass
    if replace_underscore:
        feature_name = feature_name.replace("_", " ")
    return feature_name