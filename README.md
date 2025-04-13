# 🧠 Automated Customer Review Analyzer

This project provides an end-to-end NLP pipeline for analyzing customer product reviews. It includes:

- Sentiment classification using BERT.
- Unsupervised product clustering using Sentence-BERT and HDBSCAN.
- GPT-powered summarization and insights generation.
- A Flask web interface for easy interaction and PDF export.

---

## 🚀 Project Workflow

### ✅ Task 1 – Sentiment Classification

This task classifies customer reviews into three sentiment categories:

- **Positive**
- **Neutral**
- **Negative**

#### Steps:
1. **Mapped star ratings to sentiment labels**  
   - 1–2 → Negative, 3 → Neutral, 4–5 → Positive
2. **Text cleaning**  
   - Lowercasing, removing HTML tags, special characters, etc.
3. **Class balancing**  
   - Oversampling done **before** train-test split.
4. **Tokenization**  
   - Used `bert-base-uncased` tokenizer.
5. **Model**  
   - Fine-tuned `BertForSequenceClassification` with 3 output classes.
6. **Deployment**  
   - Model saved and used later to classify new review text.

---

### ✅ Task 2 – Product Clustering

This task groups similar products into meaningful meta-categories.

#### Steps:
1. **Text Preprocessing**
   - Clean product names and categories using regex.
   - Removed noise, digits, symbols, and applied lemmatization.
2. **Text Embedding**
   - Used `all-MiniLM-L6-v2` from SentenceTransformers to generate embeddings.
3. **Initial Clustering (HDBSCAN)**
   - Applied HDBSCAN to automatically find dense clusters.
4. **Cluster Refinement (BestKFinder)**
   - Used KMeans to refine clusters based on silhouette score.
5. **Visualization**
   - Used UMAP for 2D visualization (optional).
6. **Keyword Extraction**
   - (Optional) Top words from each cluster extracted for interpretation.

#### Output:
- Products labeled with `meta_category`.
- Ready for category-wise summarization and recommendation.

---

### ✅ Task 3 – GPT Summarization & Insights

For each product category, a GPT-4 model is used to generate structured blog-style insights.

#### Summary Includes:
- **Top 3 Highest-Rated Products**
  - Average rating, key strengths, and differentiators.
- **Most Common Complaints**
  - Highlighted context-aware user pain points.
- **Worst-Rated Product**
  - Review quotes and explanation.
- **Final Recommendation**
  - Best product + suggested user types.

Summaries are saved as downloadable **PDFs**.

---

### ✅ Final Stage – Deployment

- Built with **Flask**
- Allows:
  - Uploading CSV files
  - Viewing cluster titles
  - Selecting category & summary style
  - Downloading summaries as PDFs
- Loading screen and summary dropdown included

---

## 📦 Requirements

Install all dependencies using:

```bash
pip install flask pandas openai torch transformers scikit-learn sentence-transformers hdbscan umap-learn fpdf
