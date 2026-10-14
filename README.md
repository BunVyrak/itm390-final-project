# Unsupervised Topic Discovery in Cambodian Reddit Discussion

ITM 390 Machine Learning: Final Project (Group 10)

This project finds the main discussion topics in the **r/Cambodia** subreddit without any labeled data. Comments are turned into sentence embeddings with a transformer model, reduced with UMAP, grouped with HDBSCAN, and then each group is given a readable label from its TF-IDF keywords.

---

## Repository Structure

```
.
├── Unsupervised_Topic_Discovery_Cambodia_Subreddit.ipynb   # Full pipeline (main entry point)
├── data/
│   └── embedding.npy                                       # Cached sentence embeddings (20,497 × 768)
├── Documentation/
│   ├── GP_10_UnsupervisedTopic_ITM_390_..._Final_Project.pdf   # Full project report
│   └── UnsupervisedTopicDiscovery_ITM390.pdf                   # Presentation slides
├── .gitignore
└── README.md
```

---

## Architecture

```
 ┌──────────────────────┐
 │ Kaggle dataset       │  r/Cambodia comments (26,584 rows)
 │ (kagglehub download) │
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 1. Preprocessing     │  drop "[removed]" posts, clean text,
 │                      │  drop posts with ≤ 5 words, keep English only
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 2. Embedding         │  all-mpnet-base-v2 → 768-D vectors
 │                      │  L2-normalized, saved to data/embedding.npy
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 3. Dim. reduction    │  UMAP 768-D → 70-D  (for clustering)
 │                      │  UMAP  70-D →  2-D  (for plotting)
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 4. Clustering        │  HDBSCAN on 70-D vectors
 │                      │  + reassign noise to nearest centroid
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 5. Topic labeling    │  TF-IDF (1–2 grams) per cluster → top words
 │                      │  filter junk / bot clusters
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ 6. Evaluation & viz  │  Silhouette, Davies–Bouldin,
 │                      │  Calinski–Harabasz, 2-D UMAP scatter
 └──────────────────────┘
```

---

## How It Works

### 1. Data loading
The [Reddit r/Cambodia Subreddit Dataset 2026](https://www.kaggle.com/datasets/bwandowando/reddit-rcambodia-subreddit-dataset-2026) is downloaded with `kagglehub` and `cambodia_subreddit_comments.csv` is loaded. Only the `body` (comment text) column is used.

### 2. Preprocessing
- **Remove unusable rows:** comments whose body is `[removed]` (26,584 → 26,287).
- **`clean_text()` pipeline** (in order):
  1. lowercase and expand contractions (`don't` → `do not`)
  2. remove Reddit artifacts (`[deleted]`, `r/…`, `u/…`)
  3. remove URLs and markdown formatting
  4. remove AutoModerator / bot boilerplate ("I am a bot", "please read the rules", …)
  5. strip special characters and normalize whitespace
  6. drop tokens that are ≤ 2 characters, contain digits, or are custom stopwords
- **Filter short posts:** keep only comments with more than 5 words after cleaning.
- **Language filter:** `langdetect` keeps English comments only, leaving **20,497 documents**.

### 3. Sentence embeddings
Each comment is encoded with [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) into a 768-dimensional vector that captures meaning, not just word overlap. The vectors are L2-normalized and cached to `data/embedding.npy`.

### 4. Dimensionality reduction (UMAP)
Clustering in 768 dimensions works poorly, so UMAP is used twice:

| Purpose       | n_components | n_neighbors | min_dist | metric |
|---------------|--------------|-------------|----------|--------|
| Clustering    | 70           | 50          | 0.1      | cosine |
| Visualization | 2            | 15          | 0.1      | cosine |

### 5. Clustering (HDBSCAN)
HDBSCAN is density-based: it finds the number of clusters on its own and marks low-density points as noise (`-1`).

```python
hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5,
                cluster_selection_method="leaf",
                cluster_selection_epsilon=0.05, metric="euclidean")
```

HDBSCAN initially labels 63.5% of comments as noise, which is common for short, informal text. Each noise point is then **reassigned to the nearest cluster centroid** if its cosine similarity is above 0.5. This reduces the noise from 13,021 to 7,161 points.

### 6. Topic labeling (TF-IDF)
All comments in a cluster are joined into one document, and a `TfidfVectorizer` (unigrams and bigrams, `min_df=10`, `max_df=0.8`, English, custom and junk stopwords) scores the words. The top 10 words describe the topic, and the top 3 become its label (e.g. `visa | cambodia | passport`). Clusters whose keywords match a junk list (bots, moderation, image-preview URLs) are flagged and removed.

### 7. Evaluation and visualization
Clusters are scored on the 70-D UMAP space (excluding remaining noise) and plotted on the 2-D projection. The notebook also includes EDA: comment-length distribution, detected languages, and the most frequent words.

---

## Results

| Metric                         | Value     |
|--------------------------------|-----------|
| Documents clustered            | 20,497    |
| Topics found                   | 59        |
| Initial noise ratio            | 63.5%     |
| Noise after reassignment       | 7,161     |
| Silhouette score (↑ better)    | 0.2418    |
| Davies–Bouldin index (↓ better)| 1.2346    |
| Calinski–Harabasz (↑ better)   | 2564.97   |

**Example discovered topics:**

| Label                              | Theme                          |
|------------------------------------|--------------------------------|
| `cambodia \| thai \| thailand`     | Cambodia–Thailand relations    |
| `cambodia \| corruption \| peace`  | Politics & governance          |
| `visa \| passport \| arrival`      | Visas & immigration            |
| `aba \| bank \| transfer`          | Banking & payments             |
| `rent \| hotel \| month`           | Housing & cost of living       |
| `food \| restaurant \| khmer`      | Food & dining                  |
| `china \| chinese \| scam`         | Chinese investment & scam centers |
| `schools \| school \| teacher`     | Education                      |
| `sihanoukville \| kampot \| islands` | Travel destinations          |
| `hospital \| risk \| sick`         | Health care                    |

See `Documentation/` for the full report and slides.

---

## Requirements

### Software
- **Python 3.11** (tested)
- Jupyter Notebook / JupyterLab, or VS Code with the Jupyter extension
- A [Kaggle](https://www.kaggle.com/) account (to download the dataset)

### Python packages

| Package                 | Used for                              |
|-------------------------|---------------------------------------|
| `pandas`, `numpy`       | Data handling                         |
| `sentence-transformers` | MPNet sentence embeddings (installs `torch`) |
| `umap-learn` (0.5.11)   | Dimensionality reduction              |
| `hdbscan`               | Density-based clustering              |
| `scikit-learn`          | TF-IDF, normalization, metrics        |
| `langdetect`            | Language filtering                    |
| `kagglehub`             | Dataset download                      |
| `matplotlib`, `seaborn` | Visualization                         |
| `nltk`, `tensorflow` (2.19), `keras` (3.13.2) | Imported in the notebook |

### Hardware
- ~1.2 GB free disk for the Kaggle dataset download
- 8 GB+ RAM recommended
- A GPU (CUDA or Apple MPS) speeds up embedding a lot but is optional. On CPU, encoding ~20k comments takes a while.

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/BunVyrak/itm390-final-project.git
cd itm390-final-project

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn nltk langdetect kagglehub \
            sentence-transformers umap-learn==0.5.11 hdbscan \
            tensorflow==2.19 keras==3.13.2 jupyter

# 4. Authenticate with Kaggle (one of these)
#    - place kaggle.json in ~/.kaggle/, or
#    - export KAGGLE_USERNAME=... and KAGGLE_KEY=...

# 5. Launch the notebook
jupyter notebook Unsupervised_Topic_Discovery_Cambodia_Subreddit.ipynb
```

Run all cells from top to bottom. The first run downloads the dataset (~1.2 GB) and computes embeddings, which are saved to `data/embedding.npy`.

> **Note:** UMAP uses `random_state=42` for reproducibility, but results can still vary slightly across library versions and hardware.

---

## Limitations
- Only English comments are analyzed; Khmer-language discussion is excluded.
- A large share of comments are short or conversational, which leads to a high initial noise ratio.
- Topic labels are keyword-based and may need human interpretation.
- Some clusters still capture conversational style rather than subject matter (e.g. `thanks | thank | app`).
