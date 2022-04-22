import networkx as nx
import igraph as ig
import gensim
from gensim.corpora import Dictionary

from typing import Union, Iterable, Iterator, List, Optional, Set, Dict, Tuple

def read_siwo_comms(filepath: str) -> List[List[int]]:
    """
    Read in the communities found by SIWO and return list of list of ints (vertex ids)
    """
    # each line in input file is a community
    communities = []
    for line in open(filepath, "r"):
        # nodes are separated by whitespace and we want them as ints, not strings
        community = [int(node) for node in line.split()]
        communities.append(community)

    return communities

def get_term_frequency(index: int, dictionary: Dictionary):
    """
    Given a term index and a dictionary, look up the frequency of that term
    """
    return dictionary.cfs[index]

def get_degree(index: int, graph: nx.Graph):
    """
    Given a term index and graph, return the degree of the node corresponding to term
    """
    return graph.degree[str(index)]

def get_weighted_degree(index: int, graph: nx.Graph):
    """
    Given a term index and graph, return the weighted degree of the node corresponding to term
    """
    return graph.degree(weight='weight')[str(index)]

def get_internal_degree(index: int, community: List[str], graph: nx.Graph):
    """
    Given a term index and a graph, return the internal degree of node corresponding to term,
    i.e. the number of edges connecting to nodes in same community
    """
    community_subgraph = graph.subgraph(community)
    return community_subgraph.degree[str(index)]

def get_internal_weighted_degree(index: int, community: List[str], graph: nx.Graph):
    """
    Given a term index and a graph, return the internal weighted degree of node corresponding to term,
    i.e. the sum of edge strengths connecting to nodes in same community
    """
    community_subgraph = graph.subgraph(community)
    return community_subgraph.degree(weight="weight")[str(index)]

def get_embeddedness(index: int, community: List[str], graph: nx.Graph):
    """
    Given a term index and a graph, return the embeddedness of node corresponding to term,
    i.e. ratio of internal degree to total degree
    """
    internal_degree = get_internal_degree(index, community, graph)
    degree = get_degree(index, graph)
    return internal_degree / degree

def get_weighted_embeddedness(index: int, community: List[str], graph: nx.Graph):
    """
    Given a term index and a graph, return the weighted embeddedness of node corresponding to term,
    i.e. ratio of internal weighted degree to total weighted degree
    """
    internal_degree = get_internal_weighted_degree(index, community, graph)
    degree = get_weighted_degree(index, graph)
    return internal_degree / degree