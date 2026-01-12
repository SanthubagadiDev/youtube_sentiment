#  YouTube Comments Sentiment Analysis

### 🔗Live_Demo [vist the web](https://youtubesentiment-y6yx2gpdus6zzf9u5rxtdc.streamlit.app/)

##  Project Overview

In today's digital era, audience feedback is one of the most valuable resources for content creators and brands. This project presents an **end-to-end Machine Learning pipeline** to automate sentiment analysis of YouTube comments. By leveraging **Natural Language Processing (NLP)** techniques, the system processes thousands of unstructured text entries and classifies them into **Positive**, **Negative**, or **Neutral** sentiments with high accuracy.

The system is capable of handling imbalanced datasets and short, slang-filled comments common on social media platforms. This allows creators and analysts to gain actionable insights from viewer feedback efficiently.


---

## 📊 Dataset Information 
The dataset used is the **YouTube Comments Dataset**, containing thousands of real-world comments across various video categories.

**Source:** https://www.kaggle.com/datasets/atifaliak/youtube-comments-dataset/data

**Key Features:**
- **Total Records:** Thousands of real-world comments.
- **Key Features:** Raw text comments and associated metadata.
- **Sentiment (Target)**: Categorized sentiment label (**Positive**, **Negative**, **Neutral**).

---

##  Tools & Technologies Used

- **Python:** Main programming language
- **Pandas, NumPy:** Data handling
- **Scikit-learn:** Machine Learning models
- **NLTK:** NLP preprocessing
- **Matplotlib, Seaborn:** Visualization
- **TF-IDF Vectorizer:** Text feature extraction

---

## NLP & Preprocessing Pipeline
Processing social media text requires specialized steps to clean and structure the data for modeling:

1. **Data Cleaning**: Removing URLs, special characters, emojis, and HTML tags.  
2. **Tokenization**: Breaking sentences into individual words.  
3. **Stopword Removal**: Eliminating common words (e.g., "is", "the", "and") that carry little emotional weight.  
4. **Lemmatization**: Reducing words to their root form (e.g., "running" → "run").  
5. **Vectorization**: Converting text into numerical features using **TF-IDF** or **Bag of Words**.

---

## 📈 Machine Learning Models
Several classification algorithms were evaluated to determine which handles the nuances of internet slang and short comments most effectively:

| Model                       | Accuracy |
|-------------------------------|----------|
| K-Nearest Neighbors (KNN) | 80.44 |
| Decision Tree (Before Tuning) | 81.2 |
| Multinomial Naive Bayes       | 89    | 
| Logistic Regression           | 87     | 


---

## Model Evolution & Performance
**The Challenge: Class Imbalance**
**Multinomial Naive Bayes** achieved high accuracy **(89%)**. However, due to class **imbalance in the target data**, the model was biased toward the majority class and failed to predict minority sentiments correctly.
**The Solution: Logistic Regression**
We shifted to Logistic Regression, which proved more robust in managing imbalanced text features. After fine-tuning the decision thresholds and hyperparameters, the model achieved superior predictive reliability.


---

## Key Insights
- **Text length:** Negative sentiments often correlate with shorter, more aggressive bursts of text, while positive feedback tends to be more descriptive.

- **Keywords:** Strong indicators like "amazing," "thanks," and "helpful" drove positive classifications, while specific technical complaints often flagged "neutral" or "negative" tones.

- **Model Robustness:** Logistic Regression's ability to provide probability estimates allowed us to better calibrate the model against the imbalanced nature of the dataset.
