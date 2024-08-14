# twitter-classifier

## About
Twitter sentiment analysis with 6 supervised learning models:
- XGBoost
- Random Forest
- Bag-of-words Naive Bayes Classifier w/ Add-α Smoothing
- K-Nearest Neighbors (KNN)
- Linear Support Vector Machine (SVM)
- Logistic Regression w/ ElasticNet Regularization

I trained and tested the models on a random subset of 5000 Tweets from a labeled [dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) with 1.6 million Tweets and their corresponding sentiments (negative or positive).

I used [GloVe](https://nlp.stanford.edu/projects/glove/) 25-dimension Twitter vectors for my word embeddings. Increasing the dimensionality of the vectors would increase model performances at the expense of additional computational effort.