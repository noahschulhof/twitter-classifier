import math
from collections import defaultdict
from sklearn.model_selection import train_test_split

class NaiveBayesClassifier:

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

        self.classes = self.train_data['target'].unique()
        self.class_words = {}

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
    

    def predict(self, data):
        return data['text'].map(self.predict_doc)