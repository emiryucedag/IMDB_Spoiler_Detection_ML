#  IMDB Spoiler Detection with LightGBM

> **Automated spoiler detection in movie reviews using a hybrid approach of NLP and Metadata features.**

##  Project Overview

This project aims to automatically classify IMDB movie reviews as **"Spoiler"** or **"Non-Spoiler"**. Spoilers significantly degrade the user experience on review platforms, and manual moderation at scale is inefficient.

We propose a **hybrid modeling approach** that combines textual features (TF-IDF) with structural metadata (e.g., review length, genre) using the **LightGBM** gradient boosting algorithm.

* **Objective:** Robust binary classification of spoiler content.
* **Challenge:** The dataset is highly imbalanced (**~26% spoilers**), making standard accuracy metrics misleading.
* **Solution:** A LightGBM model optimized for **Recall** and **PR-AUC**, utilizing `class_weight='balanced'` and threshold optimization.

---

## 📊 Dataset

We utilized the **IMDB Spoiler Dataset** (available on Kaggle).

* **Total Reviews:** ~573,000
* **Class Distribution:**
    * 🔴 Spoiler: 26%
    * 🟢 Non-Spoiler: 74%

### Preprocessing Strategy
1.  **Text Cleaning:** Removal of HTML tags, special characters, and noise.
2.  **Anti-Leakage Splitting:** We used `GroupShuffleSplit` based on `movie_id`.
    * *Why?* To ensure that reviews from the same movie do not appear in both the training and test sets, preventing the model from memorizing specific movie plot points (Data Leakage).

---

## 🛠️ Methodology

### 1. Hybrid Feature Engineering
We moved beyond simple text classification by engineering a hybrid feature set:
* **Textual Features:** TF-IDF Vectorization (Top 5,000 words, Unigrams + Bigrams).
* **Metadata Features:**
    * **Review Length:** Word count and character count. *(Hypothesis: Spoiler reviews tend to be longer and more detailed).*
    * **Genre:** One-Hot Encoded movie genres (e.g., Action, Thriller).

### 2. Model Architecture
* **Algorithm:** LightGBM (Light Gradient Boosting Machine).
* **Handling Imbalance:** `class_weight='balanced'` was used to penalize misclassifying the minority class.
* **Optimization:**
    * Hyperparameter tuning via **RandomizedSearchCV**.
    * **Threshold Tuning** to maximize the F1-Score on the test set.

---

## 📈 Experimental Results

Our model achieved robust performance, significantly outperforming the random baseline (0.26 PR-AUC).

| Metric | Score | Description |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.77** | Indicates good discrimination capability between classes. |
| **Recall** | **0.71** | **Critical Metric:** High success rate in catching actual spoilers. |
| **PR-AUC** | **0.57** | Primary metric for imbalanced data (Baseline: 0.26). |
| **Accuracy** | 0.70 | (Secondary metric due to class imbalance). |

###  Key Findings
* **Feature Importance:** As hypothesized during EDA, **review length (`word_count`)** proved to be one of the strongest predictors for spoilers, alongside specific keywords like *"killed"*, *"ending"*, and *"dies"*.
* **Robustness:** Regularization (L1/L2) and Early Stopping successfully prevented overfitting.

---



