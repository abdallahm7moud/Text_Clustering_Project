from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
import numpy as np


def extract_tfidf_features(data, max_features=2000, ngram_range=(1, 3), stop_words='english'):
    """
    Converts text documents into numerical TF-IDF features and returns a dense matrix.
    Allows customization of n-gram range, stop words.

    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(data)
    tfidf_matrix_dense = tfidf_matrix.toarray()  # Convert sparse matrix to dense
    return tfidf_matrix_dense, vectorizer                        
                                                            


def extract_word2vec_features(data, vector_size=1000, window=25, min_count=1):
    """
    Converts text documents into Word2Vec embeddings, averaging word vectors for each document.
  
    """
    sentences = [text.split() for text in data]
    

    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    
    # Average word vectors for each document
    embeddings = np.array([
        np.mean([model.wv[word] for word in words if word in model.wv] or [np.zeros(vector_size)], axis=0)
        for words in sentences
    ])
    return embeddings
