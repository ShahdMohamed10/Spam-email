##Project Overview

This project builds a machine learning model to classify emails as Spam or Not Spam using classical NLP techniques and supervised learning algorithms.
The goal is to compare different models and understand why some perform better on text data.

The project is implemented in a Jupyter Notebook and focuses on:

Text preprocessing

Feature extraction using NLP techniques

Training and evaluating ML models

Comparing model performance

Models Used

Logistic Regression

Multinomial Naive Bayes

#Observation:
Multinomial Naive Bayes slightly outperformed Logistic Regression, which is expected since it is well-suited for sparse word-frequency features commonly used in text classification tasks like spam detection.

##Dataset

Email dataset labeled as Spam or Ham (Not Spam)

The dataset contains raw email text used for training and testing

(Dataset source can be added here if public)

##Text Preprocessing

The following preprocessing steps were applied:

Lowercasing text

Removing punctuation and special characters

Tokenization

Stopword removal

Feature extraction using:

Bag of Words (BoW) / TF-IDF (depending on your notebook)

##Model Evaluation

The models were evaluated using standard classification metrics:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

##Key Insight:
Multinomial Naive Bayes performs better because it naturally models word frequency distributions and handles sparse, high-dimensional text data efficiently.

##Technologies Used

Python

Jupyter Notebook

Scikit-learn

Pandas

NumPy

NLTK / Sklearn text tools

##How to Run the Project

Clone the repository:

git clone https://github.com/your-username/email-spam-classification.git


Navigate to the project directory:

cd email-spam-classification


Open the notebook:

jupyter notebook


Run all cells in order.



👩‍💻 Author

Shahd Mohamed
Aspiring Machine Learning Engineer

