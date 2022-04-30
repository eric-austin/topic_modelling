import datasets
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from typing import Union, Tuple, List

def load_20_newsgroups(
    subset: str = "all", 
    with_labels: bool = False
) -> Union[List[str], Tuple[List[str], List[int]]]:
    """
    Load the 20 newsgroups dataset and return as list of strings where each string is the
    entire document.
    Can select "all", "train", or "test" subsets.
    Can return with or without topic/class labels.
    """

    newsgroups = fetch_20newsgroups(subset=subset, remove=("headers", "footers", "quotes"))
    texts = [" ".join(text.split()).lower() for text in newsgroups.data]
    if with_labels:
        labels = list(newsgroups.target)
        return texts, labels
    else:
        return texts

def load_reuters(
    subset: str = "all",
    with_labels: bool = False
) -> Union[List[str], Tuple[List[str], List[int]]]:
    """
    Load the reuters dataset and return as list of strings where each string is entire document.
    Can select "all", "train", or "test" subsets.
    Can return with or without topic/class labels.
    """
    raw_reuters = datasets.load_dataset("reuters21578", "ModLewis")

    if subset == "train":
        df = pd.DataFrame.from_dict(raw_reuters["train"])
    elif subset == "test":
        df = pd.DataFrame.from_dict(raw_reuters["test"])
    elif subset == "all":
        train_df = pd.DataFrame.from_dict(raw_reuters["train"])
        test_df = pd.DataFrame.from_dict(raw_reuters["test"])
        df = pd.concat([train_df, test_df])

    df["all_text"] = df["title"] + ". " + df["text"]
    texts = [" ".join(text.split()).lower() for text in list(df["all_text"])]

    if not with_labels:
        return texts
    else:
        labels = list(df["topics"])
        return texts, labels