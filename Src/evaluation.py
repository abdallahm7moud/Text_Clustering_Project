from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.metrics.cluster import contingency_matrix

def calculate_silhouette_score(X, labels):
    """
    Calculates the Silhouette Score for clustering.
    
    """
    return silhouette_score(X, labels)

def purity_score(y_true, y_pred):
    """
    Calculates the purity score for evaluating clustering performance.
    
    """
    # Compute contingency matrix (confusion matrix)
    matrix = contingency_matrix(y_true, y_pred)
    
    # Calculate the purity score
    purity = np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)
    
    return purity

