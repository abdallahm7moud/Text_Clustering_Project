import pandas as pd
from sklearn.datasets import fetch_20newsgroups

def load_dataset():
    """
    Load one of two datasets based on user choice:
    
    """

    # print("\nLoading Wikipedia People Dataset...")
    # documents = pd.read_csv('Data/people_wiki.csv')['text']
    
    print("\nLoading 20 Newsgroups Dataset...")
    categories = ['talk.religion.misc', 'comp.graphics', 'sci.space']
    dataset = fetch_20newsgroups(subset='all', categories=categories,
                        shuffle=False, remove=('headers', 'footers', 'quotes'))
    documents = dataset.data
    true_labels = dataset.target
    print(f"\nLoaded {len(documents)} documents.")
    
    return dataset, documents, true_labels
