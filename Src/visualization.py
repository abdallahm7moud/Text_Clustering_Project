import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram

def plot_clusters(X, labels, use_tsne=False, title='Cluster Visualization', figsize=(8, 6), save_path=None):
    """
    Visualize clusters in 2D using PCA or t-SNE with Seaborn.
    Optionally save the plot to a file.
    
    """
    if use_tsne:
        X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)
        plot_title = f'{title} (t-SNE)'
    else:
        X_embedded = PCA(n_components=2).fit_transform(X)
        plot_title = f'{title} (PCA)'

    plt.figure(figsize=figsize)
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=labels, palette='viridis', s=50)
    plt.title(plot_title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='Cluster Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Cluster plot saved to {save_path}")
    
    plt.show()


def plot_dendrogram(X, method='ward', title='Hierarchical Clustering Dendrogram', figsize=(10, 7), save_path=None):
    """
    Perform hierarchical clustering and plot a dendrogram.
    Optionally save the dendrogram to a file.
    
    """
    Z = linkage(X, method=method)

    plt.figure(figsize=figsize)
    dendrogram(Z)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Dendrogram saved to {save_path}")
    
    plt.show()
