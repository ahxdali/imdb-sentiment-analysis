# imdb-sentiment-analysis
A basic NLP project classifying IMDb movie reviews using Python and Naive Bayes.

This project uses Natural Language Processing (NLP) techniques to classify movie reviews from the IMDb dataset as **positive** or **negative**. It demonstrates text preprocessing, feature extraction using TF-IDF, and model training using a Naive Bayes classifier.

## 📊 Dataset
- Source: [IMDb Dataset of 50K Movie Reviews on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- 50,000 labeled reviews (25,000 positive, 25,000 negative)

## 🧠 Project Objectives
- Preprocess and clean textual data
- Convert text into numerical features using TF-IDF vectorization
- Train a classification model using Multinomial Naive Bayes
- Evaluate model performance with classification metrics
- Visualize review sentiment trends using word clouds and confusion matrix

## 🛠️ Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK (or TextBlob)
- Matplotlib, Seaborn
- WordCloud

## 🚀 How to Run
1. Clone the repository or open the notebook in [Google Colab](https://colab.research.google.com/)
2. Install required packages:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn wordcloud nltk
    ```
3. Download the dataset from Kaggle and upload it as `IMDB Dataset.csv`
4. Run the notebook to see preprocessing, model training, and visualizations

## 📈 Model Performance
- Model: Multinomial Naive Bayes
- TF-IDF feature extraction
- Accuracy: ~85% on the test set
- Evaluated with precision, recall, F1-score, and confusion matrix

## 📷 Visualizations
- Confusion matrix of predictions
- Word clouds for positive and negative reviews

## 📝 Results Summary
This basic NLP project successfully demonstrates how to classify text data with minimal preprocessing and standard ML tools. The model performs well and could be improved further using deep learning, ensemble models, or custom tokenization pipelines.

## 📁 File Structure
├── IMDB_Sentiment_Analysis.ipynb
├── IMDB Dataset.csv (not included – download from Kaggle)
└── README.md
## 📌 Future Improvements
- Use a more advanced model (e.g., Logistic Regression, SVM, or BERT)
- Build a simple web interface for review input and sentiment prediction
- Apply cross-validation and hyperparameter tuning

## 👤 Author
Ahad Ali  
[GitHub Profile](https://github.com/ahxdali)  
