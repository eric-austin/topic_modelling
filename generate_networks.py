"""
This program generates a series of networks according to one text preprocessing configuration.
"""

import argparse
import pickle
from time import time
from copy import deepcopy

# import community_utils
# import data_utils
import network_creation
import preprocessing

parser = argparse.ArgumentParser(description="Generate a series of term co-occurrence networks from corpus preprocessed as specified.")
parser.add_argument("-n", "--ner", help="flag for named entity recognition", type=int,
                    choices=[0,1], default=0)
parser.add_argument("-f", "--filter", help="level of filtering for parts of speech",
                    type=int, choices=[0,1,2,3], default=0)
parser.add_argument("-p", "--phrases", help="whether and what type of phrases to include",
                    choices=["none", "default", "npmi"], default="none")
parser.add_argument("-pt", "--phrase_threshold", help="threshold for filtering significant phrases",
                    type=float, default=0.0)

args = parser.parse_args()
print(args)

# load the datasets

with open("./text_datasets/20newsgroups_train.txt", "r") as f:
    newsgroups_train = f.read().split("\n")
with open("./text_datasets/20newsgroups_test.txt", "r") as f:
    newsgroups_test = f.read().split("\n")

with open("./text_datasets/reuters_train.txt", "r") as f:
    reuters_train = f.read().split("\n")
with open("./text_datasets/reuters_test.txt", "r") as f:
    reuters_test = f.read().split("\n")

with open("./text_datasets/wikitext.txt", "r") as f:
    wiki_text = f.read().split('\n')

# need to specify entity types for NER
ent_types = ["EVENT", "FAC", "GPE", "LOC", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"]
if args.ner == 0:
    use_ner = False
else:
    use_ner = True

# depending on level of pos filtering we will have different pos tags
# filter 0 means no filtering
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

# apply preprocessing
t0 = time()
newsgroups_train_docs = list(nlp.pipe(newsgroups_train))
t1 = time()
print(f"20 NG train processed in {t1 - t0} seconds")

t0 = time()
newsgroups_test_docs = list(nlp.pipe(newsgroups_test))
t1 = time()
print(f"20 NG test processed in {t1 - t0} seconds")

t0 = time()
reuters_train_docs = list(nlp.pipe(reuters_train))
t1 = time()
print(f"Reuters train processed in {t1 - t0} seconds")

t0 = time()
reuters_test_docs = list(nlp.pipe(reuters_test))
t1 = time()
print(f"Reuters test processed in {t1 - t0} seconds")

t0 = time()
wiki_text_docs = list(nlp.pipe(wiki_text))
t1 = time()
print(f"Wikitext processed in {t1 - t0} seconds")

del newsgroups_train
del newsgroups_test
del reuters_train
del reuters_test
del wiki_text


print("NLP pipeline done.")

# get the both the docs and sentences as lists of tokens
tokenized_ng_train_docs = list(preprocessing.tokenize_docs(newsgroups_train_docs, lowercase=True, sentences=False))
tokenized_ng_train_sents = list(preprocessing.tokenize_docs(newsgroups_train_docs, lowercase=True, sentences=True))
tokenized_ng_test_docs = list(preprocessing.tokenize_docs(newsgroups_test_docs, lowercase=True, sentences=False))

tokenized_rt_train_docs = list(preprocessing.tokenize_docs(reuters_train_docs, lowercase=True, sentences=False))
tokenized_rt_train_sents = list(preprocessing.tokenize_docs(reuters_train_docs, lowercase=True, sentences=True))
tokenized_rt_test_docs = list(preprocessing.tokenize_docs(reuters_test_docs, lowercase=True, sentences=False))

tokenized_wiki_docs = list(preprocessing.tokenize_docs(wiki_text_docs, lowercase=True, sentences=False))

del newsgroups_train_docs
del newsgroups_test_docs
del reuters_train_docs
del reuters_test_docs
del wiki_text_docs

# detect and apply phrases if selected
if not (args.phrases == "none"):
    t0 = time()
    # detect phrases
    ng_phrases, tokenized_ng_train_docs, ng_phrase_models = preprocessing.detect_phrases(tokenized_ng_train_docs,
                                                      num_iterations=2,
                                                      scoring_method=args.phrases,
                                                      threshold=args.phrase_threshold,
                                                      min_count=None)

    rt_phrases, rt_train_docs, rt_phrase_models = preprocessing.detect_phrases(tokenized_rt_train_docs,
                                                      num_iterations=2,
                                                      scoring_method=args.phrases,
                                                      threshold=args.phrase_threshold,
                                                      min_count=None)

    # apply phrases to sentences and test docs as well
    for model in ng_phrase_models:
        tokenized_ng_train_sents = model[tokenized_ng_train_sents]
    tokenized_ng_train_sents = [[token.replace(" ", "_") for token in sent] for sent in tokenized_ng_train_sents]

    for model in ng_phrase_models:
        tokenized_ng_test_docs = model[tokenized_ng_test_docs]
    tokenized_ng_test_docs = [[token.replace(" ", "_") for token in doc] for doc in tokenized_ng_test_docs]

    tokenized_wiki_ng_docs = deepcopy(tokenized_wiki_docs)
    for model in ng_phrase_models:
        tokenized_wiki_ng_docs = model[tokenized_wiki_ng_docs]
    tokenized_wiki_ng_docs = [[token.replace(" ", "_") for token in doc] for doc in tokenized_wiki_ng_docs]

    for model in rt_phrase_models:
        tokenized_rt_train_sents = model[tokenized_rt_train_sents]
    tokenized_rt_train_sents = [[token.replace(" ", "_") for token in sent] for sent in tokenized_rt_train_sents]

    for model in rt_phrase_models:
        tokenized_rt_test_docs = model[tokenized_rt_test_docs]
    tokenized_rt_test_docs = [[token.replace(" ", "_") for token in doc] for doc in tokenized_rt_test_docs]

    tokenized_wiki_rt_docs = deepcopy(tokenized_wiki_docs)
    for model in rt_phrase_models:
        tokenized_wiki_rt_docs = model[tokenized_wiki_rt_docs]
    tokenized_wiki_rt_docs = [[token.replace(" ", "_") for token in doc] for doc in tokenized_wiki_rt_docs]

    t1 = time()
    print(f"Phrases detected in {t1 - t0} seconds")
# still want to make sure spaces are replaced with _ 
else:
    tokenized_ng_train_docs = [[token.replace(" ", "_") for token in doc] for doc in tokenized_ng_train_docs]
    tokenized_ng_train_sents = [[token.replace(" ", "_") for token in sent] for sent in tokenized_ng_train_sents]
    tokenized_ng_test_docs = [[token.replace(" ", "_") for token in doc] for doc in tokenized_ng_test_docs]

    tokenized_rt_train_docs = [[token.replace(" ", "_") for token in doc] for doc in tokenized_rt_train_docs]
    tokenized_rt_train_sents = [[token.replace(" ", "_") for token in sent] for sent in tokenized_rt_train_sents]
    tokenized_rt_test_docs = [[token.replace(" ", "_") for token in doc] for doc in tokenized_rt_test_docs]

    tokenized_wiki_ng_docs = deepcopy(tokenized_wiki_docs)
    tokenized_wiki_ng_docs = [[token.replace(" ", "_") for token in doc] for doc in tokenized_wiki_ng_docs]

    tokenized_wiki_rt_docs = deepcopy(tokenized_wiki_docs)
    tokenized_wiki_rt_docs = [[token.replace(" ", "_") for token in doc] for doc in tokenized_wiki_rt_docs]

del tokenized_wiki_docs

# create vocabulary and dictionary objects for filtering and filter
t0 = time()
ng_vocab, ng_dictionary = preprocessing.create_vocabulary_and_dictionary(tokenized_ng_train_docs, min_threshold=None)
tokenized_ng_train_sents = preprocessing.filter_tokenized_docs_with_vocab(tokenized_ng_train_sents, ng_vocab)
tokenized_ng_train_docs = preprocessing.filter_tokenized_docs_with_vocab(tokenized_ng_train_docs, ng_vocab)
tokenized_ng_test_docs = preprocessing.filter_tokenized_docs_with_vocab(tokenized_ng_test_docs, ng_vocab)
tokenized_wiki_ng_docs = preprocessing.filter_tokenized_docs_with_vocab(tokenized_wiki_ng_docs, ng_vocab)

rt_vocab, rt_dictionary = preprocessing.create_vocabulary_and_dictionary(tokenized_rt_train_docs, min_threshold=None)
tokenized_rt_train_sents = preprocessing.filter_tokenized_docs_with_vocab(tokenized_rt_train_sents, rt_vocab)
tokenized_rt_train_docs = preprocessing.filter_tokenized_docs_with_vocab(tokenized_rt_train_docs, rt_vocab)
tokenized_rt_test_docs = preprocessing.filter_tokenized_docs_with_vocab(tokenized_rt_test_docs, rt_vocab)
tokenized_wiki_rt_drts = preprocessing.filter_tokenized_docs_with_vocab(tokenized_wiki_rt_docs, rt_vocab)

# filter out empty docs and sentences
tokenized_ng_train_sents = [sent for sent in tokenized_ng_train_sents if len(sent) > 0]
tokenized_ng_train_docs = [doc for doc in tokenized_ng_train_docs if len(doc) > 0]
tokenized_ng_test_docs = [doc for doc in tokenized_ng_test_docs if len(doc) > 0]
tokenized_wiki_ng_docs = [doc for doc in tokenized_wiki_ng_docs if len(doc) > 0]

tokenized_rt_train_sents = [sent for sent in tokenized_rt_train_sents if len(sent) > 0]
tokenized_rt_train_docs = [doc for doc in tokenized_rt_train_docs if len(doc) > 0]
tokenized_rt_test_docs = [doc for doc in tokenized_rt_test_docs if len(doc) > 0]
tokenized_wiki_rt_docs = [doc for doc in tokenized_wiki_rt_docs if len(doc) > 0]

t1 = time()
print(f"Vocabulary created and docs filtered in {t1 - t0} seconds")

# create networks for a variety of parameters
t0 = time()
ng_sentence_nb = network_creation.SentenceNetworkBuilder(tokenized_ng_train_sents, 
                                                         ng_dictionary)
ng_window5_nb = network_creation.WindowNetworkBuilder(tokenized_ng_train_sents, 
                                                     ng_dictionary, 
                                                     5)
ng_window10_nb = network_creation.WindowNetworkBuilder(tokenized_ng_train_sents, 
                                                     ng_dictionary, 
                                                     10)

rt_sentence_nb = network_creation.SentenceNetworkBuilder(tokenized_rt_train_sents, 
                                                         rt_dictionary)
rt_window5_nb = network_creation.WindowNetworkBuilder(tokenized_rt_train_sents, 
                                                     rt_dictionary, 
                                                     5)
rt_window10_nb = network_creation.WindowNetworkBuilder(tokenized_rt_train_sents, 
                                                     rt_dictionary, 
                                                     10)

for t in [0, 2]:
    ng_sentence_nb.save_network(f"./networks/ng_sentence_count_{t}.txt", type="default", threshold=t)
    ng_window5_nb.save_network(f"./networks/ng_window5_count_{t}.txt", type="default", threshold=t)
    ng_window10_nb.save_network(f"./networks/ng_window10_count_{t}.txt", type="default", threshold=t)

for t in [0.0, 0.35]:
    ng_sentence_nb.save_network(f"./networks/ng_sentence_npmi_{t}.txt", type="npmi", threshold=t)
    ng_window5_nb.save_network(f"./networks/ng_window5_npmi_{t}.txt", type="npmi", threshold=t)
    ng_window10_nb.save_network(f"./networks/ng_window10_npmi_{t}.txt", type="npmi", threshold=t)

for t in [0, 2]:
    rt_sentence_nb.save_network(f"./networks/rt_sentence_count_{t}.txt", type="default", threshold=t)
    rt_window5_nb.save_network(f"./networks/rt_window5_count_{t}.txt", type="default", threshold=t)
    rt_window10_nb.save_network(f"./networks/rt_window10_count_{t}.txt", type="default", threshold=t)

for t in [0.0, 0.35]:
    rt_sentence_nb.save_network(f"./networks/rt_sentence_npmi_{t}.txt", type="npmi", threshold=t)
    rt_window5_nb.save_network(f"./networks/rt_window5_npmi_{t}.txt", type="npmi", threshold=t)
    rt_window10_nb.save_network(f"./networks/rt_window10_npmi_{t}.txt", type="npmi", threshold=t)

t1 = time()
print(f"Networks generated and saved in {t1 - t0} seconds")

# save necessary objects for topic modelling
master_object = dict()
master_object["ng_train"] = tokenized_ng_train_docs
master_object["ng_test"] = tokenized_ng_test_docs
master_object["ng_wiki"] = tokenized_wiki_ng_docs
master_object["rt_train"] = tokenized_rt_train_docs
master_object["rt_test"] = tokenized_rt_test_docs
master_object["rt_wiki"] = tokenized_wiki_rt_docs
master_object["ng_dict"] = ng_dictionary
master_object["rt_dic"] = rt_dictionary

with open("./master_object.obj", "wb") as f:
    pickle.dump(master_object, f)
