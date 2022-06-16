import argparse
import pickle
import network_creation
import preprocessing
import community_utils
from time import time
import networkx as nx
import igraph as ig

parser = argparse.ArgumentParser(description="Preprocess documents and use CT to find topics")
parser.add_argument("-n", "--ner", help="flag for named entity recognition", type=int,
                    choices=[0,1], default=0)
parser.add_argument("-f", "--filter", help="level of filtering for parts of speech",
                    type=int, choices=[0,1,2,3], default=0)
parser.add_argument("-p", "--phrases", help="whether and what type of phrases to include",
                    choices=["none", "default", "npmi"], default="none")
parser.add_argument("-pt", "--phrase_threshold", help="threshold for filtering significant phrases",
                    type=float, default=0.0)
parser.add_argument("-ew", "--edge_weight", help="type of edge weight",
                    choices=["count", "npmi"], default="npmi")
parser.add_argument("-wt", "--weight_threshold", help="threshold for edge weights",
                    type=float, default=0)
parser.add_argument("-cd", "--cd_algorithm", help="which community detection algorithm to use",
                    choices=["leiden", "wt"], default="leiden")
parser.add_argument("-rp", "--resolution_parameter", help="resolution parameter to use for leiden algorithm",
                    type=float, default=1.0)
parser.add_argument("-d", "--dataset", help="filepath to dataset of documents")


args = parser.parse_args()
# print(args)

with open(args.dataset, "r") as f:
    docs = f.read().split("\n")

# need to specify entity types for NER
ent_types = ["EVENT", "FAC", "GPE", "LOC", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"]
if args.ner == 0:
    use_ner = False
else:
    use_ner = True

if args.filter == 0:
    pos_types = None
# filter 1 means we want to keep adjectives, adverbs, nouns, proper nouns, and verbs
elif args.filter == 1:
    pos_types = ["ADJ", "ADV", "NOUN", "PROPN", "VERB"]
# filter 2 eliminates adverbs and verbs
elif args.filter == 2:
    pos_types = ["ADJ", "NOUN", "PROPN"]
# filter 3 is just nouns and proper nouns
elif args.filter == 3:
    pos_types = ["NOUN", "PROPN"]

# create filter configuration dict
filter_dict = {"filter_short": True,
              "filter_stopwords": True,
              "filter_numbers": True,
              "filter_punct": True,
              "filter_websites": True,
              "filter_emails": True,
              "filter_not_wordlike": True,
              "pos_filters": pos_types}

# create preprocessing pipeline
nlp = preprocessing.create_pipeline(detect_sentences=True,
                                    detect_entities=use_ner,
                                    entity_types=ent_types,
                                    filter_config=filter_dict)

print("Preprocessing documents...")
t0 = time()
docs = list(nlp.pipe(docs))


tokenized_docs = list(preprocessing.tokenize_docs(docs, lowercase=True, sentences=False))
tokenized_sents = list(preprocessing.tokenize_docs(docs, lowercase=True, sentences=True))

if not (args.phrases == "none"):
    phrases, tokenized_docs, phrase_models = preprocessing.detect_phrases(tokenized_docs,
                                                      num_iterations=2,
                                                      scoring_method=args.phrases,
                                                      threshold=args.phrase_threshold,
                                                      min_count=None)
    for model in phrase_models:
        tokenized_sents = model[tokenized_sents]
    tokenized_sents = [[token.replace(" ", "_") for token in sent] for sent in tokenized_sents]
else:
    tokenized_docs = [[token.replace(" ", "_") for token in doc] for doc in tokenized_docs]
    tokenized_sents = [[token.replace(" ", "_") for token in sent] for sent in tokenized_sents]

vocab, dictionary = preprocessing.create_vocabulary_and_dictionary(tokenized_docs, min_threshold=None)
tokenized_sents = preprocessing.filter_tokenized_docs_with_vocab(tokenized_sents, vocab)
tokenized_sents = [sent for sent in tokenized_sents if len(sent) > 0]
t1 = time()
print(f"Preprocessing completed in {t1 - t0} seconds")

print("Generating network...")
t0 = time()
sentence_nb = network_creation.SentenceNetworkBuilder(tokenized_sents, dictionary)
sentence_nb.save_network(f"./network.txt", type=args.edge_weight, threshold=args.weight_threshold)
nx_g = nx.read_weighted_edgelist("./network.txt")
ig_g = ig.Graph.from_networkx(nx_g)
t1 = time()
print(f"Network generated in {t1 - t0} seconds")

print("Finding topic communities...")
t0 = time()
if args.cd_algorithm == 'leiden':
    comms = ig_g.community_leiden(resolution_parameter=args.resolution_parameter,
                                weights='weight', objective_function='modularity')
else:
    comms = ig_g.community_walktrap(weights='weight').as_clustering()
t1 = time()
print(f"Topics found in {t1 - t0} seconds")

print("Sorting topics...")
t0 = time()
comms = [[int(ig_g.vs[node]["_nx_name"]) for node in comm] for comm in comms if len(comm) > 2]
for comm in comms:
    c = [str(node) for node in comm]
    comm.sort(key=lambda node: community_utils.get_internal_weighted_degree(node, c, nx_g), reverse=True)
topics = [[dictionary[node] for node in comm] for comm in comms]
t1 = time()
print(f"Topics sorted in {t1 - t0} seconds")

with open("./topics.txt", "w") as f:
    lines = []
    for topic in topics:
        line = " ".join(topic)
        lines.append(line)
    f.write("\n".join(lines))

print("Full topics saved to topics.txt")
for topic in topics:
    print(topic[:10])