from collections import defaultdict
from gensim.corpora import Dictionary
from typing import Union, Iterable, Iterator, List, Optional, Set
import math


# class NetworkBuilder:
#     """
#     Class for constructing a word co-occurrence network from a series of tokenized documents.
#     """

#     def __init__(
#             self,
#             dictionary: Optional[Dictionary] = None
#     ) -> None:
#         """
#         Initialize object with an empty defaultdict for counting co-occurrences. If a vocab
#         Dictionary is provided then we will be able to convert tokens to ints and have the network
#         node ids be ints. Strings as node ids is problematic but if tokenized docs already in int form
#         then no dictionary is needed.
#         """
#         self.dictionary = dictionary
#         self.edge_dict = defaultdict(int)
#         self.freq_dict = defaultdict(int)

#     def __call__(
#             self,
#             tokenized_texts: Union[Iterable[Iterable[str]], Iterable[Iterable[int]]],
#             use_window: bool = False,
#             window: Optional[int] = None
#     ) -> None:
#         """
#         Calling object on tokenized texts will count up the co-occurrences given the sliding
#         window size and create a dictionary mapping edges to co-occurrence counts ie. weights
#         """
#         # need to go through each doc in the tokenized texts
#         for doc in tokenized_texts:
#             for token in doc:
#                 self.freq_dict[self.dictionary.token2id[token]] += 1
#             # if not using sliding window then set window size to length of doc
#             if not use_window:
#                 window = len(doc)
#             # iterate through all tokens (except very last) for first node
#             for i in range(len(doc) - 1):
#                 # iterate through remaining docs in window to count co-occurrences
#                 for j in range(i+1, min(i+window, len(doc))):
#                     # get the id for each token
#                     if self.dictionary is None:
#                         node_1 = doc[i]
#                         node_2 = doc[j]
#                     else:
#                         node_1 = self.dictionary.token2id[doc[i]]
#                         node_2 = self.dictionary.token2id[doc[j]]
#                     # check that edge isn't a loop
#                     if node_1 != node_2:
#                         # create an ordered edge (since direction doesn't matter)
#                         edge = (min(node_1, node_2), max(node_1, node_2))
#                         self.edge_dict[edge] += 1

#     def save_network(
#             self,
#             filepath: str,
#             threshold: int = 0,
#             weights: bool = True,
#             mode: str = "default"
#     ) -> None:
#         """
#         Save the network as an edge list file where each line is the two nodes plus optionally
#         an edge weight separated by tabs \t
        
#         Can save edge weights as raw counts, log of counts, or normalized pointwise mutual information.
#         """
#         lines = []
#         if weights:
#             for edge, count in self.edge_dict.items():
#                 if mode == "log":
#                     count = math.log(count)
#                 elif mode == "npmi":
#                     total_words = self.dictionary.num_pos
#                     freq_u = self.dictionary.cfs[edge[0]]
#                     freq_v = self.dictionary.cfs[edge[1]]
#                     prob_u = freq_u / total_words
#                     prob_v = freq_v / total_words
#                     prob_uv = count / total_words
#                     try:
#                         count = math.log(prob_uv / (prob_u * prob_v)) / -math.log(prob_uv)
#                     except ValueError:
#                         count = -1
#                 if count > threshold:
#                     lines.append(str(edge[0]) + "\t" +
#                                  str(edge[1]) + "\t" + str(count))
#         else:
#             for edge in self.edge_dict.keys():
#                 lines.append(str(edge[0]) + "\t" + str(edge[1]))

#         with open(filepath, 'w') as f:
#             f.write("\n".join(lines))

class SentenceNetworkBuilder:
    """
    Class for constructing a word co-occurrence network from a series of tokenized documents.
    """

    def __init__(
        self, 
        sentences: Iterable[Iterable[str]],
        dictionary: Dictionary
    ) -> None:
        # initialize default dicts for counting token frequencies and edge counts
        self.total_tokens = 0
        self.token_freqs = defaultdict(int)
        self.edge_counts = defaultdict(int)

        for sentence in sentences:
            # eliminate duplicates
            sentence = list(set(sentence))
            # increase token counts
            for token in sentence:
                self.total_tokens += 1
                self.token_freqs[dictionary.token2id[token]] += 1
            # count co-occurrences
            for i in range(len(sentence) - 1):
                for j in range(i + 1, len(sentence)):
                    # create ordered edge (direction doesn't matter)
                    u = dictionary.token2id[sentence[i]]
                    v = dictionary.token2id[sentence[j]]
                    edge = (min(u, v), max(u, v))
                    self.edge_counts[edge] += 1
    
    def save_network(
        self,
        filepath: str,
        type: str = "default",
        threshold: Union[int, float] = 0
    ) -> None:
        # save weighted edgelist
        lines = []
        for edge, count in self.edge_counts.items():
            if type == "npmi":
                prob_u = self.token_freqs[edge[0]] / self.total_tokens
                prob_v = self.token_freqs[edge[1]] / self.total_tokens
                prob_uv = count / self.total_tokens
                try:
                    count = math.log(prob_uv / (prob_u * prob_v)) / -math.log(prob_uv)
                except ZeroDivisionError:
                    count = -1
            if count > threshold:
                lines.append(str(edge[0]) + "\t" + str(edge[1]) + "\t" + str(count))

        with open(filepath, 'w') as f:
            f.write("\n".join(lines))



class WindowNetworkBuilder:
    """
    Class for constructing a word co-occurrence network from a series of tokenized documents.
    """

    def __init__(
        self, 
        docs: Iterable[Iterable[str]],
        dictionary: Dictionary,
        window: int = 5
    ) -> None:
        # initialize default dicts for counting token frequencies and edge counts
        self.total_tokens = 0
        self.token_freqs = defaultdict(int)
        self.edge_counts = defaultdict(int)

        for doc in docs:
            for i in range(len(doc)):
                # get window
                window_tokens = doc[i:min(i+window, len(doc))]
                # only count first token
                u = dictionary.token2id[window_tokens[0]]
                self.total_tokens += 1
                self.token_freqs[u] += 1
                # remove duplicates and count co-occurrence
                vs = list(set(window_tokens[1:]))
                for token in vs:
                    v = dictionary.token2id[token]
                    if u != v:
                        edge = (min(u, v), max(u, v))
                        self.edge_counts[edge] += 1

            
    
    def save_network(
        self,
        filepath: str,
        type: str = "default",
        threshold: Union[int, float] = 0
    ) -> None:
        # save weighted edgelist
        lines = []
        for edge, count in self.edge_counts.items():
            if type == "npmi":
                prob_u = self.token_freqs[edge[0]] / self.total_tokens
                prob_v = self.token_freqs[edge[1]] / self.total_tokens
                prob_uv = count / self.total_tokens
                try:
                    count = math.log(prob_uv / (prob_u * prob_v)) / -math.log(prob_uv)
                except ZeroDivisionError:
                    count = -1
            if count > threshold:
                lines.append(str(edge[0]) + "\t" + str(edge[1]) + "\t" + str(count))

        with open(filepath, 'w') as f:
            f.write("\n".join(lines))
