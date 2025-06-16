# Netflix Movies and TV Shows Clustering

## Description

This project focuses on uncovering meaningful patterns within the Netflix content library using unsupervised machine learning techniques. By clustering movies and TV shows based on textual features such as genre, description, and cast, the model helps in content recommendation, segmentation, and catalog management. This project employs NLP preprocessing, TF-IDF vectorization, dimensionality reduction, and clustering algorithms to group similar shows/movies, providing valuable insights into Netflix’s global content offering.

---

## Table of Contents
- [Project Introduction](#project-introduction)
- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preparation](#data-preparation)
- [Modeling Phase](#modeling-phase)
- [Evaluation Metric](#evaluation-metric)
- [Conclusion](#conclusion)
- [Project Structure](#project-structure)

## Project Introduction

With thousands of titles available on Netflix, it becomes essential to categorize and recommend content intelligently. This project clusters similar titles by analyzing their attributes like genre, description, cast, and release year. The insights can be used to improve user experience through enhanced recommendation systems or content segmentation.

## Dataset Description

The dataset, sourced from [Kaggle's Netflix Movies and TV Shows dataset](https://www.kaggle.com/shivamb/netflix-shows), contains over 8,800 records with fields such as title, director, cast, genre, country, release year, duration, and description. Our focus is on textual features for clustering content meaningfully.

## Exploratory Data Analysis (EDA)

Key findings from EDA:
- **Type Distribution:** ~70% content are movies, rest are TV shows.
- **Top Genres:** Dramas, Documentaries, and International Movies dominate the catalog.
- **Country Analysis:** Majority of content originates from the USA, followed by India and UK.
- **Release Year Trends:** A sharp increase in new titles post-2015.
- **Cast Frequency:** Certain actors and directors appear across multiple titles, influencing content themes.

## Data Preparation

- Combined relevant text fields: *title*, *director*, *cast*, *listed_in*, and *description* into a single feature.
- Preprocessing included:
  - Lowercasing
  - Removing stopwords
  - Lemmatization using `nltk`
- Converted textual data into vectors using **TF-IDF vectorization**.
- Applied **Truncated SVD** for dimensionality reduction.

## Modeling Phase

Implemented multiple clustering algorithms:
- **KMeans Clustering**: Tested for various cluster counts using the Elbow method and Silhouette Score.
- **DBSCAN**: Density-based clustering for handling noise and outliers.
- **Agglomerative Clustering**: Hierarchical approach for understanding nested relationships between clusters.

## Evaluation Metric

As unsupervised learning lacks ground truth labels, the following evaluation metrics were used:
- **Silhouette Score** to measure intra-cluster cohesion and inter-cluster separation.
- **Davies–Bouldin Index** for validating cluster compactness and separation.
- Visualization using **TSNE** and **2D PCA plots** to inspect cluster separability.

## Conclusion

The **KMeans algorithm** with TF-IDF features and dimensionality reduction gave the most interpretable clusters. The resulting groupings show that shows/movies with similar genres, themes, and casts tend to cluster together, validating the effectiveness of text-based clustering in content recommendation.

## Project Structure

```
.
├── data/                       # Dataset CSV files
├── notebooks/                  # Jupyter notebooks for EDA and modeling
│   └── Netflix_Clustering.ipynb
├── models/                     # Saved cluster models and artifacts
├── src/                        # Python scripts for preprocessing and clustering
│   ├── text_preprocessing.py
│   ├── tfidf_vectorization.py
│   ├── clustering.py
│   └── evaluation.py
├── README.md                   # Project documentation (this file)
├── requirements.txt            # Dependencies list
└── results/                    # Cluster visualizations and metrics
```

## Notable Techniques Used

- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Truncated SVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
- [KMeans Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Agglomerative Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- Visualization: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Plotly](https://plotly.com/)

## Libraries and Technologies

- **Python**
- **Pandas, NumPy** (data manipulation)
- **Scikit-learn** (ML modeling)
- **NLTK** (NLP preprocessing)
- **Matplotlib, Seaborn, Plotly** (visualizations)
- **Jupyter Notebooks** (interactive development)

## Dataset Source

- [Kaggle: Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)

---

This project provides a foundation for content-based recommendation systems, showcasing how NLP and unsupervised learning can enhance the digital media experience.
