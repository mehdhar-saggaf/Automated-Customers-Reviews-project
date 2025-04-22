import numpy as np
import hdbscan
import torch
import re
import string
import nltk
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# ======= Download NLTK Resources =======
nltk.download('stopwords')
nltk.download('wordnet')
# ======= Main Function: cluster_products =======
def cluster_products(df):
    # Step 1: Clean Text
    df['text_for_clustering'] = df['name'].fillna('') + ' ' + df['categories'].fillna('')
    df['text_for_clustering'] = df['text_for_clustering'].apply(clean_text)
    # Step 2: Generate Embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    if torch.cuda.is_available():
        model = model.to('cuda')
        print(":white_check_mark: Model moved to GPU")
    else:
        print(":zap: No GPU detected, using CPU")
    texts = df['text_for_clustering'].dropna().tolist()
    texts = [t.strip() for t in texts if t.strip()]
    embeddings = model.encode(texts, batch_size=16, convert_to_numpy=True, show_progress_bar=True)
    # Step 3: HDBSCAN Clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=150, prediction_data=True)
    hdbscan_labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    n_noise = sum(hdbscan_labels == -1)
    print(f"\n:large_blue_circle: [HDBSCAN] Number of clusters found: {n_clusters}")
    print(f":zap: Number of noise points (outliers): {n_noise}")
    if n_clusters == 0:
        # All noise, assign -1
        df.loc[:, "cluster"] = -1
        df.loc[:, "meta_category"] = ["Noise"] * len(hdbscan_labels)
        return df
    # Step 4: Improve Clustering with BestKFinder
    k_finder = BestKFinder()
    k_finder.fit(embeddings, hdbscan_labels)
    if k_finder.get_labels() is not None:
        best_labels = k_finder.get_labels()
        df = df.head(len(best_labels))
        df['cluster'] = best_labels
        print(f":trophy: Best K selected: {k_finder.get_best_k()}")
        print(f":chart_with_upwards_trend: Best Silhouette Score: {k_finder.get_best_score():.2f}")
    else:
        # fallback
        df = df.head(len(hdbscan_labels))
        df['cluster'] = hdbscan_labels
    # :white_check_mark::white_check_mark: Add meta_category so Flask does not crash
    df['meta_category'] = ["Category {}".format(c) if c >= 0 else "Noise" for c in df['cluster']]
    return df
# ======= BestKFinder Class =======
class BestKFinder:
    def __init__(self, k_min=None, k_max=None):
        self.k_min = k_min
        self.k_max = k_max
        self.best_k = None
        self.best_score = None
        self.best_labels = None
    def fit(self, X, hdbscan_labels):
        mask = hdbscan_labels != -1
        X_filtered = X[mask]
        if X_filtered.shape[0] == 0:
            return
        X_normalized = normalize(X_filtered)
        n_clusters_hdbscan = len(np.unique(hdbscan_labels[mask]))
        if self.k_min is None or self.k_max is None:
            self.k_min = 2
            self.k_max = max(2, n_clusters_hdbscan // 2)
        best_score = -1
        best_k = None
        best_labels = None
        for k in range(self.k_min, self.k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            current_labels = kmeans.fit_predict(X_normalized)
            score = silhouette_score(X_normalized, current_labels, metric='cosine')
            if score > best_score:
                best_k = k
                best_score = score
                best_labels = current_labels
        self.best_k = best_k
        self.best_score = best_score
        self.best_labels = best_labels
    def get_best_k(self):
        return self.best_k
    def get_best_score(self):
        return self.best_score
    def get_labels(self):
        return self.best_labels
# ======= Helper Function: Clean Text =======
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'allnew', '', text)
    text = re.sub(r'\d+gb', '', text)
    text = re.sub(r'display', '', text)
    text = re.sub(r'wifi', '', text)
    text = re.sub(r'hd', '', text)
    text = re.sub(r'back', '', text)
    text = re.sub(r'gb', '', text)
    text = re.sub(r'ip', '', text)
    text = re.sub(r'aaa', '', text)
    text = re.sub(r'aa', '', text)
    text = re.sub(r'ad', '', text)
    text = re.sub(r'[^a-z\s,]', '', text)
    categories = [cat.strip() for cat in text.split(',') if cat.strip()]
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english') + [
        'new', 'misc', 'offer', 'offers', 'magenta',
        'fire', 'special', 'miscellaneous', 'other',
        'includes', 'including', 'amazon', 'back', 'allnew', 'wifi'
    ])
    pattern = re.compile('[%s]' % re.escape(string.punctuation))
    cleaned_categories = []
    for cat in categories:
        words = cat.split()
        filtered_words = [
            lemmatizer.lemmatize(pattern.sub('', word))
            for word in words
            if word not in stop_words and len(word) > 1
        ]
        if filtered_words:
            cleaned_categories.append(' '.join(filtered_words))
    return ', '.join(cleaned_categories)