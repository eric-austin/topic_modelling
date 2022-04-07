from collections import defaultdict
from gensim.corpora import Dictionary
from typing import Union, Iterable, Iterator, List, Optional, Set
import math


class NetworkBuilder:
    """
    Class for constructing a word co-occurrence network from a series of tokenized documents.
    """

    def __init__(
            self,
            dictionary: Optional[Dictionary] = None
    ) -> None:
        """
        Initialize object with an empty defaultdict for counting co-occurrences. If a vocab
        Dictionary is provided then we will be able to convert tokens to ints and have the network
        node ids be ints. Strings as node ids is problematic but if tokenized docs already in int form
        then no dictionary is needed.
        """
        self.dictionary = dictionary
        self.edge_dict = defaultdict(int)

    def __call__(
            self,
            tokenized_texts: Union[Iterable[Iterable[str]], Iterable[Iterable[int]]],
            use_window: bool = False,
            window: Optional[int] = None
    ) -> None:
        """
        Calling object on tokenized texts will count up the co-occurrences given the sliding
        window size and create a dictionary mapping edges to co-occurrence counts ie. weights
        """
        # need to go through each doc in the tokenized texts
        for doc in tokenized_texts:
            # if not using sliding window then set window size to length of doc
            if not use_window:
                window = len(doc)
            # iterate through all tokens (except very last) for first node
            for i in range(len(doc) - 1):
                # iterate through remaining docs in window to count co-occurrences
                for j in range(i+1, min(i+window, len(doc))):
                    # get the id for each token
                    if self.dictionary is None:
                        node_1 = doc[i]
                        node_2 = doc[j]
                    else:
                        node_1 = self.dictionary.token2id[doc[i]]
                        node_2 = self.dictionary.token2id[doc[j]]
                    # check that edge isn't a loop
                    if node_1 != node_2:
                        # create an ordered edge (since direction doesn't matter)
                        edge = (min(node_1, node_2), max(node_1, node_2))
                        self.edge_dict[edge] += 1

    def save_network(
            self,
            filepath: str,
            threshold: int = 0,
            weights: bool = True,
            log: bool = False
    ) -> None:
        """
        Save the network as an edge list file where each line is the two nodes plus optionally
        an edge weight separated by tabs \t
        """
        lines = []
        if weights:
            for edge, count in self.edge_dict.items():
                if log:
                    count = math.log(count)
                if count > threshold:
                    lines.append(str(edge[0]) + "\t" +
                                 str(edge[1]) + "\t" + str(count))
        else:
            for edge in self.edge_dict.keys():
                lines.append(str(edge[0]) + "\t" + str(edge[1]))

        with open(filepath, 'w') as f:
            f.write("\n".join(lines))
