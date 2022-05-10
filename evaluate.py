import os
import pickle
import networkx as nx
import igraph as ig
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LdaModel
from time import time


import community_utils

def main():
    ner = "1"
    pos_filter = "0"
    phrase = "npmi"
    phrase_threshold = "0.35"

    with open("./ng1_master_object.obj", "rb") as f:
        master_object = pickle.load(f)

    ng_dict = master_object["ng_dict"]
    # rt_dict = master_object["rt_dict"]
    # bbc_dict = master_object["bbc_dict"]

    f = open("ng1_results_wtleiden.csv", "a")
    # want to go through each network, find the associated siwo communities and
    # mine the leiden communities, then evaluate all permutations
    for network in os.listdir("./ng1_networks"):
        t0 = time()
        # f = open(f"ng1_results_{network}.csv", "a")
        # break up filename to grab params for network generation
        details = network.split("_")
        train_data = details[0]
        window = details[1]
        edge_type = details[2]
        weight_threshold = details[3][:-4]

        # each network will have two associated siwo community partitions
        # siwo_a = community_utils.read_siwo_comms(f"./bbc2_siwo_comms/a_{network}")
        # siwo_g = community_utils.read_siwo_comms(f"./bbc2_siwo_comms/g_{network}")
        # # filter small comms
        # siwo_a = [comm for comm in siwo_a if len(comm) > 2]
        # siwo_g = [comm for comm in siwo_g if len(comm) > 2]

        # load network
        nx_g = nx.read_weighted_edgelist(f"./ng1_networks/{network}")
        ig_g = ig.Graph.from_networkx(nx_g)

        # use different resolution parameters for leiden
        walktrap = ig_g.community_walktrap(weights='weight').as_clustering()
        # leiden_1 = ig_g.community_leiden(weights='weight', 
        #                                 objective_function='CPM',
        #                                 resolution_parameter=1)
        # leiden_50 = ig_g.community_leiden(weights='weight', 
        #                                 objective_function='CPM',
        #                                 resolution_parameter=0.5)
        # leiden_25 = ig_g.community_leiden(weights='weight', 
        #                                 objective_function='CPM',
        #                                 resolution_parameter=0.25)
        # leiden_10 = ig_g.community_leiden(weights='weight', 
        #                                 objective_function='CPM',
        #                                 resolution_parameter=0.1)
        # filter small comms
        walktrap = [[int(ig_g.vs[node]["_nx_name"]) for node in comm] for comm in walktrap if len(comm) > 2]
        # leiden_1 = [[int(ig_g.vs[node]["_nx_name"]) for node in comm] for comm in leiden_1 if len(comm) > 2]
        # leiden_50 = [[int(ig_g.vs[node]["_nx_name"]) for node in comm] for comm in leiden_50 if len(comm) > 2]
        # leiden_25 = [[int(ig_g.vs[node]["_nx_name"]) for node in comm] for comm in leiden_25 if len(comm) > 2]
        # leiden_10 = [[int(ig_g.vs[node]["_nx_name"]) for node in comm] for comm in leiden_10 if len(comm) > 2]

        alg_params = [
            ("walktrap,4", walktrap),
        ]
        
        # now we have 2 siwo partitions and 4 leiden partitions to evaluate
        # will evaluate each with different term ranking functions
        for alg_param, partition in alg_params:
            # need the right dictionary
            dictionary = ng_dict


            # # sort by degree
            # for comm in partition:
            #     comm.sort(key=lambda node: community_utils.get_degree(node, nx_g), reverse=True)
            # # get topics as terms
            # topics = [[dictionary[node] for node in comm] for comm in partition]

            # # evaluate on train, test, and wiki with both c_v and npmi
            # for ref in ["test"]:
            #     key = train_data + "_" + ref
            #     ref_corpus = master_object[key]
            #     for coherence in ["c_v", "c_npmi"]:
            #         for topn in [5, 10, 20]:
            #             cm = CoherenceModel(topics=topics, texts=ref_corpus, dictionary=dictionary, topn=topn, coherence=coherence)
            #             score = cm.get_coherence()
            #             row = f"{train_data},{ref},{ner},{pos_filter},{phrase},{phrase_threshold},{alg_param},{window},{edge_type},{weight_threshold},degree,{coherence},{topn},{score}"
            #             f.write(row + "\n")


            # # sort by weighted degree
            # for comm in partition:
            #     comm.sort(key=lambda node: community_utils.get_weighted_degree(node, nx_g), reverse=True)
            # # get topics as terms
            # topics = [[dictionary[node] for node in comm] for comm in partition]

            # # evaluate on train, test, and wiki with both c_v and npmi
            # for ref in ["test"]:
            #     key = train_data + "_" + ref
            #     ref_corpus = master_object[key]
            #     for coherence in ["c_v", "c_npmi"]:
            #         for topn in [5, 10, 20]:
            #             cm = CoherenceModel(topics=topics, texts=ref_corpus, dictionary=dictionary, topn=topn, coherence=coherence)
            #             score = cm.get_coherence()
            #             row = f"{train_data},{ref},{ner},{pos_filter},{phrase},{phrase_threshold},{alg_param},{window},{edge_type},{weight_threshold},weighted_degree,{coherence},{topn},{score}"
            #             f.write(row + "\n")

            # # sort by internal degree
            # for comm in partition:
            #     c = [str(node) for node in comm]
            #     comm.sort(key=lambda node: community_utils.get_internal_degree(node, c, nx_g), reverse=True)
            # # get topics as terms
            # topics = [[dictionary[node] for node in comm] for comm in partition]

            # # evaluate on train, test, and wiki with both c_v and npmi
            # for ref in ["test"]:
            #     key = train_data + "_" + ref
            #     ref_corpus = master_object[key]
            #     for coherence in ["c_v", "c_npmi"]:
            #         for topn in [5, 10, 20]:
            #             cm = CoherenceModel(topics=topics, texts=ref_corpus, dictionary=dictionary, topn=topn, coherence=coherence)
            #             score = cm.get_coherence()
            #             row = f"{train_data},{ref},{ner},{pos_filter},{phrase},{phrase_threshold},{alg_param},{window},{edge_type},{weight_threshold},internal_degree,{coherence},{topn},{score}"
            #             f.write(row + "\n")

            # sort by internal weighted degree
            for comm in partition:
                c = [str(node) for node in comm]
                comm.sort(key=lambda node: community_utils.get_internal_weighted_degree(node, c, nx_g), reverse=True)
            # get topics as terms
            topics = [[dictionary[node] for node in comm] for comm in partition]

            # evaluate on train, test, and wiki with both c_v and npmi
            for ref in ["test"]:
                key = train_data + "_" + ref
                ref_corpus = master_object[key]
                for coherence in ["c_v", "c_npmi"]:
                    for topn in [5, 10, 20]:
                        cm = CoherenceModel(topics=topics, texts=ref_corpus, dictionary=dictionary, topn=topn, coherence=coherence)
                        score = cm.get_coherence()
                        row = f"{train_data},{ref},{ner},{pos_filter},{phrase},{phrase_threshold},{alg_param},{window},{edge_type},{weight_threshold},internal_weighted_degree,{coherence},{topn},{score}"
                        f.write(row + "\n")

            # # sort by embeddedness
            # for comm in partition:
            #     c = [str(node) for node in comm]
            #     comm.sort(key=lambda node: community_utils.get_embeddedness(node, c, nx_g), reverse=True)
            # # get topics as terms
            # topics = [[dictionary[node] for node in comm] for comm in partition]

            # # evaluate on train, test, and wiki with both c_v and npmi
            # for ref in ["test"]:
            #     key = train_data + "_" + ref
            #     ref_corpus = master_object[key]
            #     for coherence in ["c_v", "c_npmi"]:
            #         for topn in [5, 10, 20]:
            #             cm = CoherenceModel(topics=topics, texts=ref_corpus, dictionary=dictionary, topn=topn, coherence=coherence)
            #             score = cm.get_coherence()
            #             row = f"{train_data},{ref},{ner},{pos_filter},{phrase},{phrase_threshold},{alg_param},{window},{edge_type},{weight_threshold},embeddedness,{coherence},{topn},{score}"
            #             f.write(row + "\n")

            # # sort by weighted embeddedness
            # for comm in partition:
            #     c = [str(node) for node in comm]
            #     comm.sort(key=lambda node: community_utils.get_weighted_embeddedness(node, c, nx_g), reverse=True)
            # # get topics as terms
            # topics = [[dictionary[node] for node in comm] for comm in partition]

            # # evaluate on train, test, and wiki with both c_v and npmi
            # for ref in ["test"]:
            #     key = train_data + "_" + ref
            #     ref_corpus = master_object[key]
            #     for coherence in ["c_v", "c_npmi"]:
            #         for topn in [5, 10, 20]:
            #             cm = CoherenceModel(topics=topics, texts=ref_corpus, dictionary=dictionary, topn=topn, coherence=coherence)
            #             score = cm.get_coherence()
            #             row = f"{train_data},{ref},{ner},{pos_filter},{phrase},{phrase_threshold},{alg_param},{window},{edge_type},{weight_threshold},weighted_embeddedness,{coherence},{topn},{score}"
            #             f.write(row + "\n")
        t1 = time()
        print(f"{t1 - t0} seconds", network)
    f.close()


if __name__ == "__main__":
    main()