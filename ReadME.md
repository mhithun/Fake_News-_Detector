# Fake News Detection

A machine-learning project for detecting fake news articles.
This repository contains:

- A trained text classification model (`fake_news_model.pkl`) and TF‑IDF vectorizer (`tfidf_vectorizer.pkl`)
- A Streamlit web app (`app.py`) to classify news as real or fake
- Notebooks and data used to train and analyze the model (`Fake.csv`, `True.csv`)

---

## Features

- Classify a news article as **REAL** or **FAKE**
- Confidence score for each prediction
- Text preprocessing with:
  - Lowercasing, URL/HTML removal
  - Punctuation and digit removal
  - Stopword removal (NLTK)
  - Lemmatization (WordNet)
- Interactive Streamlit UI with:
  - Title + content inputs
  - Sample fake / real news examples
  - Styled result cards and basic metrics (confidence, word count)
- Training and analysis notebooks for:
  - Dataset exploration and cleaning
  - Traditional ML models with TF‑IDF
  - Experimental semantic search / knowledge base with `sentence-transformers` and `chromadb`

---

## Project Structure

- `app.py` – Streamlit app to run the fake news detector
- `fake_news_model.pkl` – Trained ML model for classification
- `tfidf_vectorizer.pkl` – Fitted TF‑IDF vectorizer
- `Fake.csv` – Fake news dataset (training data)
- `True.csv` – Real news dataset (training data)
- `Untitled1.ipynb` – Main training & EDA notebook (TF‑IDF + ML models)
- `Untitled.ipynb` – Experimental notebook (semantic embeddings + ChromaDB)
- `news_kb/`
  - `chroma.sqlite3` – Persisted ChromaDB knowledge base
- `ReadME.md` – Project documentation (this file)

---

## Requirements

### Core app (inference) dependencies

You mainly need:

- Python 3.8+
- `streamlit`
- `pandas`
- `numpy`
- `joblib`
- `nltk`
- (optional but common) `scikit-learn` for working with the model/vectorizer types

Install them with:

```bash
pip install streamlit pandas numpy joblib nltk scikit-learn
```

### Training / notebooks (optional)

For running the notebooks end‑to‑end you may also need:

```bash
pip install sentence-transformers chromadb transformers accelerate bitsandbytes \
           wordcloud matplotlib seaborn
```

---

## NLTK Setup

The app automatically downloads the required NLTK data (stopwords, WordNet) using:

- `stopwords`
- `wordnet`
- `omw-1.4`

This happens when you first run the app via a cached helper function.
If you see NLTK download errors, you can also download manually in a Python shell:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## Running the App

1. **Clone the repository** (or place the files in a folder).
2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv venv
   # Windows PowerShell:
   venv\Scripts\Activate.ps1
   # macOS / Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install streamlit pandas numpy joblib nltk scikit-learn
   ```

4. **Make sure model files are present** in the project root:

   - `fake_news_model.pkl`
   - `tfidf_vectorizer.pkl`

5. **Start the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

6. Open the URL shown in your terminal (usually `http://localhost:8501`).

---

## Using the App

1. Enter a **news title** and **news content** in the text fields.
2. Or expand **“Try Sample News Articles”** and load one of the predefined fake/real examples.
3. Click **“Analyze News”**.
4. The app will show:
   - Label: **FAKE NEWS DETECTED** or **LIKELY REAL NEWS**
   - Optional confidence score (if the model exposes probabilities)
   - Number of words analyzed
   - A short reminder to fact‑check important content

If the model/vectorizer files are missing, the app will show an error message and stop.

---

## Model & Dataset

### Dataset

The training dataset is stored in:

- `Fake.csv` – fake news articles
- `True.csv` – real news articles

In the notebooks, these are:

- Loaded and labeled (`0` = fake, `1` = true)
- Concatenated into a single dataframe
- Explored with:
  - Class distribution
  - Text length statistics
  - Basic visualizations (e.g., word clouds)

### Preprocessing

Key preprocessing steps (both in training and in the app):

- Convert text to lowercase
- Remove URLs and HTML tags
- Remove punctuation and digits
- Normalize whitespace
- Tokenize and:
  - Remove English stopwords
  - Lemmatize tokens using WordNet
- Join tokens back into a cleaned string

### Model

- Text features: **TF‑IDF** representation of the cleaned text
- Classifiers explored in notebooks include:
  - Logistic Regression
  - Naive Bayes
  - Random Forest
  - Gradient Boosting
  - SVM
- The final selected model is saved as `fake_news_model.pkl` along with `tfidf_vectorizer.pkl`.

The Streamlit app uses a simple pipeline:

```text
input text -> clean_text() -> vectorizer.transform() -> model.predict()
```

---

## Experimental Knowledge Base

The notebook `Untitled.ipynb` demonstrates building a simple knowledge base of **verified news**:

- Uses `sentence-transformers` (`all-mpnet-base-v2`) to embed real news text
- Stores embeddings in a persistent **ChromaDB** collection (`news_kb/chroma.sqlite3`)
- Can be used for semantic search / retrieval over real news articles (not yet integrated into `app.py` UI)

---

## Limitations & Disclaimer

- The model is trained on a specific dataset and may not generalize perfectly to all news sources or domains.
- Predictions are **best-effort estimates**, not definitive truth.
- Do not rely on this tool as the sole method of verifying news.
- Always cross-check important information with trusted, authoritative sources.

---

## Possible Improvements

Some ideas for future work:

- Add advanced models (e.g., transformer‑based classifiers).
- Integrate the semantic knowledge base into the Streamlit UI for explanation / evidence.
- Support multiple languages.
- Add model monitoring and evaluation on new data.
- Containerize the app (Docker) and deploy to a cloud platform.

---

## License

Specify your preferred license here (e.g. MIT, Apache 2.0) and include the corresponding `LICENSE` file in the repository.