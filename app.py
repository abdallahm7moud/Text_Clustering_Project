import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Src.preprocessing import preprocess_data
from Src.feature_extraction import extract_word2vec_features
from sklearn.decomposition import PCA

# Load the saved model
kmeans_model = joblib.load("Models/kmeans_model.pkl")
vectorizer = joblib.load("Models/vectorizer.pkl")

# Function to predict cluster
def predict_cluster(text):
    preprocessed_text = preprocess_data(text, lemmatization=True)
    features = vectorizer.transform([preprocessed_text])
    # pca = PCA(n_components=2)
    # features = pca.fit_transform(features) 
    cluster = kmeans_model.predict(features)[0]
    return cluster

# Function to process dataset and visualize clusters
def process_and_visualize(df, text_column):
    st.write("### Data Preview")
    st.write(df.head())
    
    # Preprocess text data
    df['processed_text'] = df[text_column].apply(lambda x: preprocess_data(str(x), lemmatization=True))
    features = vectorizer.transform(df['processed_text'])
    
    # Predict clusters
    df['Cluster'] = kmeans_model.predict(features)
    
    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features.toarray())
    df['PCA1'], df['PCA2'] = reduced_features[:, 0], reduced_features[:, 1]
    
    # Plot clusters
    st.write("### Cluster Visualization")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['Cluster'], palette='viridis')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Clusters Visualization")
    st.pyplot(plt)
    
    return df

# Streamlit UI
st.title("Document Clustering App")

# Text input for single document clustering
user_input = st.text_area("Enter text for clustering:")
if st.button("Cluster Document"):
    cluster = predict_cluster(user_input)
    st.write(f"Predicted Cluster: {cluster}")

# File upload section
st.write("### Upload Dataset for Clustering")
uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=["csv", "txt"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_table(uploaded_file)
    text_column = st.selectbox("Select Text Column", df.columns)
    if st.button("Process and Visualize Clusters"):
        df = process_and_visualize(df, text_column)
        st.write("### Clustered Data Preview")
        st.write(df[['processed_text', 'Cluster']].head())
