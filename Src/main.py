from load_dataset import load_dataset
from preprocessing import preprocess_data
from feature_extraction import extract_tfidf_features, extract_word2vec_features
from clustering import kmeans_clustering, hierarchical_clustering, lda_clustering
from evaluation import calculate_silhouette_score, purity_score
from visualization import plot_clusters, plot_dendrogram
from sklearn.decomposition import PCA
import joblib
import pandas as pd


def run_pipeline():
    """
    Main function to run the document clustering pipeline.
    
    """
    
    print("Running Document Clustering Pipeline...")

    # Step 1: Load Dataset or Use Input Text
    dataset, documents, true_labels = load_dataset()
    
    
    # Step 2: Preprocess the Text Data
    print("Preprocessing text data...")
    preprocessed_texts = [preprocess_data(doc, lemmatization=True) for doc in documents]
    print("Preprocessing completed.\n")
    print("===============================================\n")
    
    
    # Step 3: Feature Extraction Using TF-IDF
    print("Using TF-IDF features...")
    features, vectorizer = extract_tfidf_features(preprocessed_texts)
    print("Feature extraction completed.\n")
    print("===============================================\n")
    
    # Save the vectorizer
    joblib.dump(vectorizer, "Models/vectorizer.pkl")
    print("vectorizer saved successfully.\n")
    print("===============================================\n")
    
    
    # Step 4: Apply PCA for dimensionality reduction
    # print("\nApplying PCA for dimensionality reduction...")
    # pca = PCA(n_components=2)
    # features = pca.fit_transform(features)
    # print(f"Reduced feature dimensions: {features.shape[1]}\n")
    # print("===============================================\n")
    
    
    # Step 5: Apply Clustering Algorithm 
    print("\nApplying KMeans Clustering...")
    kmeans_model, cluster_labels = kmeans_clustering(features)
    print("KMeans clustering completed.\n")
    print("===============================================\n")

    # Save KMeans model
    joblib.dump(kmeans_model, "Models/kmeans_model.pkl")
    print("KMeans model saved successfully.\n")
    print("===============================================\n")

    df = pd.DataFrame({"Document": documents, "Cluster": true_labels})
    df.to_csv("Results/kmeans_clusters.csv", index=False)
    
        
    # Evaluate clusters for KMeans
    print("\nCalculating Silhouette Score...")
    silhouette = calculate_silhouette_score(features, cluster_labels)
    print(f"Silhouette Score: {silhouette:.4f}\n")
    print("===============================================\n")
    
    print("\nCalculating Purity Score...")
    purity = purity_score(true_labels, cluster_labels)
    print(f"Purity Score: {purity:.4f}\n")
    print("===============================================\n")
    
    print("Visualizing clusters with PCA...")
    plot_clusters(features, cluster_labels, title="KMeans Clustering with PCA", save_path="Results/kmeans_Clustering.png")
    print("Visualization completed.\n")
    print("===============================================\n")
    
    print("Pipeline finished.\n")
    print("===============================================\n")




if __name__ == "__main__":
        run_pipeline()