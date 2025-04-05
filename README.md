# Text_Clustering_Project

A comprehensive project for unsupervised clustering of text documents using machine learning techniques. This project includes data preprocessing, feature extraction, clustering using K-Means, and interactive visualization through a Streamlit web app.

---

## 📁 Project Structure

```
Text_Clustering_Project/
│
├── Data/                   # Contains raw data files (e.g., .rar archives)
│
├── Models/                 # Pretrained models and vectorizers
│   ├── kmeans_model.pkl
│   └── vectorizer.pkl
│
├── Notebooks/              # Jupyter Notebooks for EDA and analysis
│   └── Text_EDA.ipynb
│
├── Results/                # Plots and results from clustering
│   └── kmeans_Clustering.png
│
├── Src/                    # Source Python modules
│   ├── clustering.py
│   ├── evaluation.py
│   ├── feature_extraction.py
│   ├── load_dataset.py
│   ├── main.py
│   ├── preprocessing.py
│   └── visualization.py
│
├── app.py                  # Streamlit app for deployment
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignored files
└── README.md               # Project documentation
```

---

## 📌 Features

- 📄 Load and preprocess real-world text data
- 🧹 Tokenization, stopword removal, and vectorization (TF-IDF)
- 📊 K-Means clustering for unsupervised document grouping
- 📈 Visualizations of clusters using PCA/t-SNE
- 🌐 Streamlit app for interactive exploration

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Text_Clustering_Project.git
cd Text_Clustering_Project
```

### 2. Install dependencies

We recommend using a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📂 Data

The dataset is stored in the `Data/` folder as a compressed `.rar` file. Please extract it before running the project.

---

## 🧠 Models

- `kmeans_model.pkl`: Pretrained K-Means clustering model
- `vectorizer.pkl`: TF-IDF vectorizer used for feature extraction

---

## 📓 Notebooks

- `Text_EDA.ipynb`: Exploratory Data Analysis (EDA) for text data

---

## 🛠️ Modules

- `load_dataset.py`: Load and split the dataset
- `preprocessing.py`: Clean and preprocess text
- `feature_extraction.py`: Convert text to numerical features (TF-IDF)
- `clustering.py`: Apply K-Means clustering
- `evaluation.py`: Evaluate clustering performance
- `visualization.py`: Visualize clusters using dimensionality reduction

---

## 📊 Results

Visualizations and clustering results are saved in the `Results/` folder.

---

## 🌐 Deployment

This project includes a Streamlit app (`app.py`) for interactive exploration of clusters and model predictions.

---

## 🙌 Acknowledgements

- Scikit-learn
- NLTK
- Streamlit
- 20 Newsgroups Dataset (from sklearn.datasets)

---
