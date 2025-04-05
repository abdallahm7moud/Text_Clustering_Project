from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation
from feature_extraction import extract_tfidf_features

def kmeans_clustering(X, n_clusters=3):
    """
    Performs KMeans clustering on input data X.
    Returns the trained KMeans model and cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans, kmeans.labels_

def hierarchical_clustering(X, n_clusters=3):
    """
    Performs Agglomerative (Hierarchical) Clustering on input data X.
    Returns the trained AgglomerativeClustering model and cluster labels.
    """
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative.fit(X)
    return agglomerative, agglomerative.labels_

def lda_clustering(X, n_topics=3, evaluate_perplexity=True):
    """
    Performs Latent Dirichlet Allocation (LDA) on input data X.
    Returns the LDA model, topic distribution for each document, and optionally the perplexity.
    """
    text_features = extract_tfidf_features(X)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_distributions = lda.fit_transform(text_features) 
    topic_labels = topic_distributions.argmax(axis=1)  

    if evaluate_perplexity:
        perplexity = lda.perplexity(text_features)
        return lda, topic_distributions, topic_labels, perplexity, text_features

    return lda, topic_distributions, topic_labels, text_features
