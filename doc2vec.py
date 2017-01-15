import logging
import sys
from random import shuffle

from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from preprocessor import ParseTweets

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)
import numpy as np


class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


class Doc2vecLearner:
    def __init__(self):
        self.model = None
        self.pt = ParseTweets()

    def learn_doc2vec(self, sources=None, model_save='model/twitter1.d2v', epoch=1, **kwargs):
        if sources is None:
            sources = {'data/corpus/all_english1.txt': 'DATA'}
        print(model_save)
        sentences = TaggedLineSentence(sources)
        model = Doc2Vec(workers=16, **kwargs)
        model.build_vocab(sentences.to_array())
        for num in range(epoch):
            print("EPOCH " + str(num))
            model.train(sentences.sentences_perm())

        print("Model save")
        model.save(model_save)
        self.model = model

    def load_model(self, model_file='model/twitter1.d2v'):
        self.model = Doc2Vec.load(model_file)

    def doctext_to_vec(self, doc):
        return None if self.model is None else self.model.infer_vector(self.pt.tokenize_one_line(doc))

    def doctext_to_word2vec(self, doc):
        words, _ = self.pt.tokenize_one_line(doc, True)
        return np.mean(np.array([self.model[word] for word in words if word in self.model]), axis=0)

    def doctext_to_word2vec_with_tfidf(self, doc, tf_idf_model, features):
        words, no_token = self.pt.tokenize_one_line(doc, True)
        tf_idf_vec = tf_idf_model.transform_tfidf([no_token])
        # print([tf_idf_vec[0, features.index(word)] for word in words if word in features])
        return np.mean(np.array([tf_idf_vec[0, features.index(word)] * self.model[word]
                                 for word in words if word in self.model and word in features]), axis=0)


if __name__ == "__main__":
    a = Doc2vecLearner()
    a.learn_doc2vec(model_save='model/twitter2.d2v', epoch=10)
