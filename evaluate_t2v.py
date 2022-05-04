from top2vec import Top2Vec
import pickle
from time import time
from gensim.models.coherencemodel import CoherenceModel

def main():
    ner = "1"
    pos_filter = "3"
    phrase = "npmi"
    phrase_threshold = "0.35"

    with open("./bbc_master_object.obj", "rb") as f:
        master_object = pickle.load(f)

    # ng_dict = master_object["ng_dict"]
    bbc_dict = master_object["bbc_dict"]

    f = open("bbc_t2v_results.csv", "a")

    # first lets evaluate LDA
    # let's use 20NG first
    # t0 = time()
    # corpus = [" ".join(text) for text in master_object["ng_train"]]
    # t2v = Top2Vec(corpus, speed="learn")
    # topic_words, _, _ = t2v.get_topics()
    # for ref in ["test", "train"]:
    #     key = "ng_" + ref
    #     ref_corpus = master_object[key]
    #     for coherence in ["c_v", "c_npmi"]:
    #         for topn in [5, 10, 20]:
    #             cm = CoherenceModel(topics=topic_words, texts=ref_corpus, dictionary=ng_dict, topn=topn, coherence=coherence)
    #             score = cm.get_coherence()
    #             row = f"ng,{ref},{ner},{pos_filter},{phrase},{phrase_threshold},t2v,na,na,na,na,na,{coherence},{topn},{score}"
    #             f.write(row + "\n")
    # t1 = time()
    # print(f"t2v on 20 NG finished in {t1 - t0} seconds")

    # now we'll do reuters
    t0 = time()
    corpus = [" ".join(text) for text in master_object["bbc_train"]]
    t2v = Top2Vec(corpus, speed="learn")
    topic_words, _, _ = t2v.get_topics()
    for ref in ["test"]:
        key = "bbc_" + ref
        ref_corpus = master_object[key]
        for coherence in ["c_v", "c_npmi"]:
            for topn in [5, 10, 20]:
                cm = CoherenceModel(topics=topic_words, texts=ref_corpus, dictionary=bbc_dict, topn=topn, coherence=coherence)
                score = cm.get_coherence()
                row = f"bbc,{ref},{ner},{pos_filter},{phrase},{phrase_threshold},t2v,na,na,na,na,na,{coherence},{topn},{score}"
                f.write(row + "\n")
    t1 = time()
    print(f"t2v on BBC finished in {t1 - t0} seconds")

    f.close()

if __name__ == "__main__":
    main()