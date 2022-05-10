import pickle
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LdaModel
from time import time
from top2vec import Top2Vec
import numpy as np
import community_utils
import network_creation
import networkx as nx
import igraph as ig
import random

def main():
    with open("./ng1_master_object.obj", "rb") as f:
            master_object = pickle.load(f)

    dictionary = master_object["ng_dict"]
    corpus = [dictionary.doc2bow(text) for text in master_object["ng_train"]]

    lda_times = []
    lda_cvs = []
    lda_npmis = []

    for i in range(10):
        random.shuffle(corpus)
        t0 = time()
        model = LdaModel(corpus, num_topics=20, iterations=2000)
        t1 = time()
        lda_times.append(t1 - t0)

        cm = CoherenceModel(model=model, texts=master_object["ng_test"], dictionary=dictionary, topn=10, coherence="c_v")
        lda_cvs.append(cm.get_coherence())
        cm = CoherenceModel(model=model, texts=master_object["ng_test"], dictionary=dictionary, topn=10, coherence="c_npmi")
        lda_npmis.append(cm.get_coherence())

        print(f"Done LDA {i}")

    print("LDA")
    print(f"Times: {np.mean(lda_times)} +/- {np.std(lda_times)}")
    print(f"CVs: {np.mean(lda_cvs)} +/- {np.std(lda_cvs)}")
    print(f"NPMIs: {np.mean(lda_npmis)} +/- {np.std(lda_npmis)}")

    t2v_times = []
    t2v_cvs = []
    t2v_npmis = []
    corpus = [" ".join(text) for text in master_object["ng_train"]]

    for i in range(10):
        random.shuffle(corpus)
        t0 = time()
        model = Top2Vec(corpus, speed="learn", workers=1)
        t1 = time()
        t2v_times.append(t1 - t0)
        topic_words, _, _ = model.get_topics()

        cm = CoherenceModel(topics=topic_words, texts=master_object["ng_test"], dictionary=dictionary, topn=10, coherence="c_v")
        t2v_cvs.append(cm.get_coherence())
        cm = CoherenceModel(topics=topic_words, texts=master_object["ng_test"], dictionary=dictionary, topn=10, coherence="c_npmi")
        t2v_npmis.append(cm.get_coherence())

        print(f"Done t2v {i}")

    print("T2V")
    print(f"Times: {np.mean(t2v_times)} +/- {np.std(t2v_times)}")
    print(f"CVs: {np.mean(t2v_cvs)} +/- {np.std(t2v_cvs)}")
    print(f"NPMIs: {np.mean(t2v_npmis)} +/- {np.std(t2v_npmis)}")


    network_times = []

    for i in range(10):
        t0 = time()
        window10_nb = network_creation.WindowNetworkBuilder(master_object["ng_train"], 
                                                            dictionary, 
                                                            10)
        t1 = time()
        network_times.append(t1 - t0)

    leiden_times = []
    walktrap_times = []
    sort_times = []
    siwo_cvs = []
    siwo_npmis = []
    leiden_cvs = []
    leiden_npmis = []
    walktrap_cvs = []
    walktrap_npmis = []

    nx_g = nx.read_weighted_edgelist("./ng1_networks/ng_sentence_count_0.txt")
    ig_g = ig.Graph.from_networkx(nx_g)

    for i in range(10):
        siwo_a = community_utils.read_siwo_comms("./ng1_siwo_comms/a_ng_sentence_count_0.txt")
        siwo_a = [comm for comm in siwo_a if len(comm) > 2]

        t0 = time()
        for comm in siwo_a:
            c = [str(node) for node in comm]
            comm.sort(key=lambda node: community_utils.get_internal_weighted_degree(node, c, nx_g), reverse=True)
        t1 = time()
        sort_times.append(t1 - t0)
        topics = [[dictionary[node] for node in comm] for comm in siwo_a]

        cm = CoherenceModel(topics=topics, texts=master_object["ng_test"], dictionary=dictionary, topn=10, coherence="c_v")
        siwo_cvs.append(cm.get_coherence())
        cm = CoherenceModel(topics=topics, texts=master_object["ng_test"], dictionary=dictionary, topn=10, coherence="c_npmi")
        siwo_npmis.append(cm.get_coherence())

        t0 = time()
        leiden = ig_g.community_leiden(weights='weight', objective_function='modularity')
        t1 = time()
        leiden_times.append(t1 - t0)
        leiden = [[int(ig_g.vs[node]["_nx_name"]) for node in comm] for comm in leiden if len(comm) > 2]
        for comm in leiden:
            c = [str(node) for node in comm]
            comm.sort(key=lambda node: community_utils.get_internal_weighted_degree(node, c, nx_g), reverse=True)
        topics = [[dictionary[node] for node in comm] for comm in leiden]

        cm = CoherenceModel(topics=topics, texts=master_object["ng_test"], dictionary=dictionary, topn=10, coherence="c_v")
        leiden_cvs.append(cm.get_coherence())
        cm = CoherenceModel(topics=topics, texts=master_object["ng_test"], dictionary=dictionary, topn=10, coherence="c_npmi")
        leiden_npmis.append(cm.get_coherence())

        t0 = time()
        walktrap = ig_g.community_walktrap(weights='weight').as_clustering()
        t1 = time()
        walktrap_times.append(t1 - t0)
        walktrap = [[int(ig_g.vs[node]["_nx_name"]) for node in comm] for comm in walktrap if len(comm) > 2]
        for comm in walktrap:
            c = [str(node) for node in comm]
            comm.sort(key=lambda node: community_utils.get_internal_weighted_degree(node, c, nx_g), reverse=True)
        topics = [[dictionary[node] for node in comm] for comm in walktrap]

        cm = CoherenceModel(topics=topics, texts=master_object["ng_test"], dictionary=dictionary, topn=10, coherence="c_v")
        walktrap_cvs.append(cm.get_coherence())
        cm = CoherenceModel(topics=topics, texts=master_object["ng_test"], dictionary=dictionary, topn=10, coherence="c_npmi")
        walktrap_npmis.append(cm.get_coherence())

    print(f"Network creation: {np.mean(network_times)} +/- {np.std(network_times)}")
    print(f"Sorting: {np.mean(sort_times)} +/- {np.std(sort_times)}")
    print("Leiden")
    print(f"Times: {np.mean(leiden_times)} +/- {np.std(leiden_times)}")
    print(f"CVs: {np.mean(leiden_cvs)} +/- {np.std(leiden_cvs)}")
    print(f"NPMIs: {np.mean(leiden_npmis)} +/- {np.std(leiden_npmis)}")
    print("Walktrap")
    print(f"Times: {np.mean(walktrap_times)} +/- {np.std(walktrap_times)}")
    print(f"CVs: {np.mean(walktrap_cvs)} +/- {np.std(walktrap_cvs)}")
    print(f"NPMIs: {np.mean(walktrap_npmis)} +/- {np.std(walktrap_npmis)}")
    print("SIWO")
    print(f"CVs: {np.mean(siwo_cvs)} +/- {np.std(siwo_cvs)}")
    print(f"NPMIs: {np.mean(siwo_npmis)} +/- {np.std(siwo_npmis)}")

    tns_dict = dict()

    tns_dict["lda_times"] = lda_times
    tns_dict["lda_cvs"] = lda_cvs
    tns_dict["lda_npmis"] = lda_npmis
    tns_dict["t2v_times"] = t2v_times
    tns_dict["t2v_cvs"] = t2v_cvs
    tns_dict["t2v_npmis"] = t2v_npmis
    tns_dict["network_times"] = network_times
    tns_dict["sort_times"] = sort_times
    tns_dict["leiden_times"] = leiden_times
    tns_dict["leiden_cvs"] = leiden_cvs
    tns_dict["leiden_npmis"] = leiden_npmis
    tns_dict["walktrap_times"] = walktrap_times
    tns_dict["walktrap_cvs"] = walktrap_cvs
    tns_dict["walktrap_npmis"] = walktrap_npmis
    tns_dict["siwo_cvs"] = siwo_cvs
    tns_dict["siwo_npmis"] = siwo_npmis

    with open("tns_results.obj", "wb") as f:
        pickle.dump(tns_dict, f)

if __name__ == "__main__":
    main()