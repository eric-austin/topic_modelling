
import pickle
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LdaModel
from time import time

def main():
    ner = "1"
    pos_filter = "3"
    phrase = "npmi"
    phrase_threshold = "0.35"

    with open("./master_object.obj", "rb") as f:
        master_object = pickle.load(f)

    ng_dict = master_object["ng_dict"]
    rt_dict = master_object["rt_dic"]

    f = open("lda_results.csv", "a")

    # first lets evaluate LDA
    # let's use 20NG first
    t0 = time()
    for n_topics in [20, 50, 100]:
        corpus = [ng_dict.doc2bow(text) for text in master_object["ng_train"]]
        lda = LdaModel(corpus, num_topics=n_topics)
        for ref in ["test", "train"]:
            key = "ng_" + ref
            ref_corpus = master_object[key]
            for coherence in ["c_v", "c_npmi"]:
                for topn in [5, 10, 20]:
                    cm = CoherenceModel(model=lda, texts=ref_corpus, dictionary=ng_dict, topn=topn, coherence=coherence)
                    score = cm.get_coherence()
                    row = f"ng,{ref},{ner},{pos_filter},{phrase},{phrase_threshold},lda,{n_topics},na,na,na,na,{coherence},{topn},{score}"
                    f.write(row + "\n")
    t1 = time()
    print(f"LDA on 20 NG finished in {t1 - t0} seconds")

    # now we'll do reuters
    t0 = time()
    for n_topics in [20, 50, 100]:
        corpus = [rt_dict.doc2bow(text) for text in master_object["rt_train"]]
        lda = LdaModel(corpus, num_topics=n_topics)
        for ref in ["test", "train"]:
            key = "rt_" + ref
            ref_corpus = master_object[key]
            for coherence in ["c_v", "c_npmi"]:
                for topn in [5, 10, 20]:
                    cm = CoherenceModel(model=lda, texts=ref_corpus, dictionary=rt_dict, topn=topn, coherence=coherence)
                    score = cm.get_coherence()
                    row = f"rt,{ref},{ner},{pos_filter},{phrase},{phrase_threshold},lda,{n_topics},na,na,na,na,{coherence},{topn},{score}"
                    f.write(row + "\n")
    t1 = time()
    print(f"LDA on Reuters finished in {t1 - t0} seconds")

    f.close()

if __name__ == "__main__":
    main()