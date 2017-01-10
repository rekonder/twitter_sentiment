from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdf:
    def __init__(self):
        self.tfidf = None
        pass

    def read_data(self, data='data/corpus/all_english1.txt'):
        with open(data, "r") as fil:
            fil.readline()
            for row in fil.readlines():
                 yield row

    def learn_tfidf(self, data='data/corpus/all_english1.txt', **kwargs):
        self.tfidf = TfidfVectorizer(**kwargs)
        self.tfidf.fit(self.read_data(data))

    def transform_tfidf(self, data):
        return None if self.tfidf is None else self.tfidf.transform(data)



