from transformers import pipeline
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def zeroshot_classify_sexualization(df, column_name, labels):

    #Initialize the zero-shot classifier
    zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = labels

    results = []

    for text in df[column_name]:
        classification = zero_shot_classifier(
            text,
            candidate_labels=candidate_labels,
            multi_label=False
        )

        best_label = classification["labels"][0]
        best_score = classification["scores"][0]

        results.append((best_label, best_score))

    df["sexualization_prediction"] = [r[0] for r in results]
    df["sexualization_confidence"] = [r[1] for r in results]

    return df

def process_and_cluster_movies(df):
    """
    Preprocess movie data, generate embeddings, and perform clustering by genre.

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame with columns 'Plot Summaries' and 'Main_genre'.

    Returns:
        pd.DataFrame: DataFrame with added cluster labels.
        dict: Cluster details with top terms per genre.
    """
    data = df.copy()
    data = data.dropna(subset=['Plot Summaries', 'Main_genre'])

    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    nlp = spacy.load("en_core_web_sm")

    #Preprocessing Function
    def preprocess_text(text):
        doc = nlp(text)
        tokens = [token.text for token in doc if token.pos_ != "PROPN" and not token.ent_type_ and not token.is_stop]
        return " ".join(tokens)

    data['Plot Summaries'] = data['Plot Summaries'].apply(preprocess_text)

    #Generate SBERT Embeddings
    def get_sbert_embeddings(texts, model):
        return model.encode(texts, show_progress_bar=True)

    #Clustering Function
    def cluster_genre_group(group):
        embeddings = get_sbert_embeddings(group['Plot Summaries'].tolist(), sbert_model)

        #TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(group['Plot Summaries']).toarray()

        #Combine Features
        combined_features = np.hstack((embeddings, tfidf_matrix))

        #Optimal Clusters
        range_n_clusters = range(2, min(10, len(group) + 1))
        silhouette_scores = []
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(combined_features)
            silhouette_avg = silhouette_score(combined_features, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        optimal_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        group['cluster'] = kmeans.fit_predict(combined_features)

        return group, tfidf_vectorizer, tfidf_matrix

    #Extract Top Terms
    def get_top_terms_per_cluster(tfidf_vectorizer, cluster_labels, tfidf_matrix, n_terms=10):
        terms = tfidf_vectorizer.get_feature_names_out()
        clusters_top_terms = {}
        for cluster in np.unique(cluster_labels):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_tfidf_mean = tfidf_matrix[cluster_indices].mean(axis=0)
            top_indices = np.argsort(cluster_tfidf_mean)[::-1][:n_terms]
            top_terms = [terms[i] for i in top_indices]
            clusters_top_terms[cluster] = top_terms
        return clusters_top_terms

    #Apply Clustering
    results = []
    genre_cluster_details = {}
    for genre, group in data.groupby('Main_genre'):
        if len(group) < 2:
            group['cluster'] = 0
            results.append(group)
            continue

        clustered_group, tfidf_vectorizer, tfidf_matrix = cluster_genre_group(group)
        results.append(clustered_group)
        cluster_labels = clustered_group['cluster']
        top_terms = get_top_terms_per_cluster(tfidf_vectorizer, cluster_labels, tfidf_matrix)
        genre_cluster_details[genre] = top_terms

    final_data = pd.concat(results)

    return final_data, genre_cluster_details
