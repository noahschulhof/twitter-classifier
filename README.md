# twitter-classifier

## About
I classified the sentiments of Tweets using 5 Supervised Classification Algorithms:
- K-Nearest Neighbors (KNN)
- Bag-of-words Naive Bayes Classifier w/ Add-α Smoothing
- Linear Support Vector Machine (SVM)
- Random Forest Classifier
- Logistic Regression w/ ElasticNet Regularization

Models were trained and tested on a random subset of 10000 Tweets from a labeled [dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) with 1.6 million Tweets and their corresponding sentiments (negative or positive).

## Preparing Input Files
After cloning the repository, raw input files can be prepared with the singular command
```bash
$ bash download.sh
```
Please note that this is only necessary if running `preprocess.ipynb`. The data used to train/test the models in `models.ipynb` is available in the `cleaned` folder.