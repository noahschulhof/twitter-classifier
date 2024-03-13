import pandas as pd
import math
from collections import defaultdict
from sklearn.model_selection import train_test_split

class NaiveBayesClassifier:

    def __init__(self, data):
        self.data = pd.read_csv(data)

        self.classes = self.data['target'].unique()
        self.class_words = {}

        self.train_data, self.test_data = train_test_split(self.data, test_size = 0.2, random_state = 1)

        self.vocabulary = set([])
        self.logprior = {}
        self.loglikelihood = {}

        
    def pretrain(self):
        class_docs = defaultdict(list)

        for _, row in self.train_data.iterrows():
            class_docs[row['target']].append(row['text'])

        n_docs = sum([len(docs) for docs in class_docs.values()])

        for c, d in class_docs.items():
            self.logprior[c] = math.log(len(d)/n_docs)
            self.class_words[c] = [word for doc in d for word in doc.split() if '@' not in word and 'http' not in word]

            for word in self.class_words[c]:
                self.vocabulary.add(word)
    
    
    def train(self, alpha):
        for word in self.vocabulary:
            for c, c_words in self.class_words.items():
                self.loglikelihood[(word, c)] = math.log((c_words.count(word) + alpha)/(len(c_words) + alpha * len(self.vocabulary)))
                
        
    def score(self, doc, c):
        return self.logprior[c] + sum([self.loglikelihood[(word, c)] for word in doc.split() if word in self.vocabulary])
                

    def predict_doc(self, doc):
        class_scores = {c: self.score(doc, c) for c in self.classes}
        
        return max(class_scores, key = class_scores.get)
    

    def predict(self, subset = 'test'):
        if subset == 'test':
            return self.test_data['text'].apply(lambda x: self.predict_doc(x))
        
        return self.train_data['text'].apply(lambda x: self.predict_doc(x))