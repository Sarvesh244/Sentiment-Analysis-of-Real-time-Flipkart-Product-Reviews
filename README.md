# 🏸 Sentiment Analysis of Flipkart Product Reviews

## 📋 Project Overview

This project aims to classify customer reviews from Flipkart as **Positive** or **Negative** and identify key pain points driving customer dissatisfaction. The specific focus is on the "YONEX MAVIS 350 Nylon Shuttle".

By analyzing the sentiment of over 8,500 reviews, this solution provides insights into product features that contribute to customer satisfaction (e.g., durability, delivery speed) versus dissatisfaction (e.g., damaged goods, fake products). The final model is deployed as a web application using **Streamlit** on **AWS EC2**.

## 📊 Dataset

The dataset consists of **8,518 real-time reviews** scraped from Flipkart.

* **Product:** YONEX MAVIS 350 Nylon Shuttle
* **Key Features:** Review text, Ratings, Reviewer Name, Date, Up/Down Votes.
* **Target Variable:** derived from `Ratings` (4-5 stars = Positive, 1-2 stars = Negative). Neutral ratings (3 stars) were excluded to focus on binary classification.

## 🛠 Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, Pickle
* **Web Framework:** Streamlit
* **Deployment:** AWS EC2 (Ubuntu instance)

## ⚙️ Workflow & Methodology

### 1. Data Preprocessing

* **Cleaning:** Removed HTML tags, special characters, and punctuation.
* **Normalization:** Applied lowercasing.
* **Stopword Logic:** carefully tuned to **preserve negation words** (e.g., "not", "don't") to ensure phrases like "not good" are correctly classified as negative.
* **Handling Nulls:** Removed empty review rows to ensure data integrity.

### 2. Feature Extraction

Experimented with multiple techniques:

* **Bag-of-Words (BoW):** Selected for the final model due to superior performance.
* **TF-IDF:** Tested for comparison.
* **N-Grams:** Used Uni-grams and Bi-grams to capture context (e.g., "waste money", "good quality").

### 3. Model Training

Trained and evaluated the following models:

* Logistic Regression (Selected Model)
* Naive Bayes
* Random Forest

**Performance:**

* **Best Model:** Logistic Regression with Bag-of-Words
* **Accuracy:** ~92%
* **F1-Score:** ~0.96

### 4. Insights & Pain Points

Analysis of model coefficients revealed major drivers of negative sentiment:

* **Product Condition:** Keywords like "damaged", "old", "dried".
* **Authenticity:** Keywords like "fake", "duplicate", "cheated".
* **Price/Value:** Keywords like "expensive", "waste".

## 🚀 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/flipkart-sentiment-analysis.git
cd flipkart-sentiment-analysis

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Run the App Locally

```bash
streamlit run app.py

```

## 🌐 Deployment

The application is deployed on an **AWS EC2 instance**.

* **Instance Type:** t2.micro (Free Tier)
* **OS:** Ubuntu Server
* **Port:** 8501 (Streamlit default)

## 📂 Directory Structure

```
├── data.csv                 # Raw dataset
├── sentiment analysis.ipynb           # Script to train and save the model
├── streamlitapp.py                   # Streamlit web application
├── sentiment_model.pkl      # Saved Machine Learning Model
├── vectorizer.pkl           # Saved CountVectorizer
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

```

## 🤝 Future Improvements

* Integrate Deep Learning models (LSTM/BERT) for potentially higher accuracy on complex sentences.
* Add a dashboard to visualize sentiment trends over time (e.g., did quality drop in a specific month?).
* Expand the dataset to include multiple product categories.

---

*Created by Sarvesh Mote*
