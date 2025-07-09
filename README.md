# imdb-sentiment-analysis
📽️ IMDb Sentiment Analysis

A basic NLP project classifying IMDb movie reviews using Python and Naive Bayes.

This project uses Natural Language Processing (NLP) techniques to classify movie reviews from the IMDb dataset as positive or negative. It demonstrates text preprocessing, feature extraction using TF-IDF, and model training using a Naive Bayes classifier.

📊 Dataset
Source: IMDb Dataset of 50K Movie Reviews on Kaggle
Link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

50,000 labeled reviews (25,000 positive, 25,000 negative)

🧠 Project Objectives
Preprocess and clean textual data

Convert text into numerical features using TF-IDF vectorization

Train a classification model using Multinomial Naive Bayes

Evaluate model performance with classification metrics

Visualize review sentiment trends using word clouds and confusion matrix

🛠️ Tools & Libraries
Python

pandas, numpy

scikit-learn

nltk

matplotlib, seaborn

wordcloud

🚀 How to Run
Clone this repository or open the notebook in Google Colab

Install required packages (in Colab, use !pip install as needed):

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud nltk
Download the dataset from Kaggle and upload it as IMDB Dataset.csv

Run all notebook cells to see preprocessing, model training, and visualizations

📈 Model Performance
Model: Multinomial Naive Bayes

Feature Extraction: TF-IDF

Accuracy: ~85% on the test set

Evaluated using:

Precision

Recall

F1-score

Confusion matrix

📷 Visualizations
Confusion matrix of predictions

Word clouds for:

Positive reviews

Negative reviews

📝 Results Summary
This beginner NLP project demonstrates how to classify sentiment in movie reviews using standard machine learning tools. The Naive Bayes model performs well and provides a strong baseline. Future enhancements may improve performance and usability.

📁 File Structure
csharp
Copy
Edit
├── IMDB_Sentiment_Analysis.ipynb  
├── IMDB Dataset.csv (not included – download from Kaggle)  
└── README.md  
📌 Future Improvements
Use a more advanced model (e.g., Logistic Regression, SVM, or BERT)

Build a simple web app (e.g., with Flask or Streamlit)

Apply cross-validation and hyperparameter tuning

Use lemmatization or more advanced NLP techniques

👤 Author
Ahad Ali
https://github.com/ahxdali
