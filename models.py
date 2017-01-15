import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from doc2vec import Doc2vecLearner
from tfidf import TfIdf
from utils import read_train_test_data, measurement, read_train_test_data_test


def basic_model(doc2vec=None, classifier=None, test=False):
    """
    Model ki uporablja za atribute TF-IDF predstavitev dokumenta
    """
    if doc2vec is None:
        doc2vec = Doc2vecLearner()
        doc2vec.load_model(model_file='model/twitter1.d2v')
    train_X, train_Y, test_X, test_Y = read_train_test_data(doc2vec)
    print("Learn")
    if classifier is None:
        classifier = LogisticRegression(class_weight='balanced')

    classifier.fit(train_X, train_Y)
    print("Predict")
    prediction = [classifier.predict(i) for i in test_X]

    print("Dev-test")
    measurement(test_Y, prediction)
    if test:
        test_X, test_Y = read_train_test_data_test(doc2vec)
        prediction = [classifier.predict(i) for i in test_X]
        print("test")
        measurement(test_Y, prediction)


def tfidf_model(tfidf=None, classifier=None, test=False):
    """
    Model ki uporablja za atribute povrečno doc2vec predstavitev besed v dokumentu
    """
    if tfidf is None:
        tfidf = TfIdf()
        tfidf.learn_tfidf(stop_words='english', max_df=0.8)
    print("Read data")
    train_X, train_Y, test_X, test_Y = read_train_test_data(tfidf, mode="tfidf")
    print("Learn")
    if classifier is None:
        classifier = LogisticRegression(class_weight='balanced')
    classifier.fit(train_X, train_Y)
    print("Predict")
    prediction = [classifier.predict(i) for i in test_X]
    print("Dev-test")
    measurement(test_Y, prediction)
    if test:
        test_X, test_Y = read_train_test_data_test(tfidf, mode="tfidf")
        prediction = [classifier.predict(i) for i in test_X]
        print("test")
        measurement(test_Y, prediction)


def basic_tfidf_model(tfidf=None, doc2vec=None, classifier=None):
    """
    Model ki uporablja za atribute doc2vec in TF-IDF predstavitev besed v dokumentu
    """
    if doc2vec is None:
        doc2vec = Doc2vecLearner()
        doc2vec.load_model(model_file='model/twitter1.d2v')
    if tfidf is None:
        print("tf-idf")
        tfidf = TfIdf()
        tfidf.learn_tfidf(stop_words='english', max_df=0.8)
    print("Read data")
    train_X, train_Y, test_X, test_Y = read_train_test_data(model1=doc2vec, model2=tfidf, mode="both")
    print("Learn")
    if classifier is None:
        classifier = LogisticRegression(class_weight='balanced')
    classifier.fit(train_X, train_Y)
    print("Predict")
    prediction = [classifier.predict(i) for i in test_X]
    print(len(prediction))
    measurement(test_Y, prediction)


def doc2vec_wordvec_model(doc2vec=None, classifier=None, test=False):
    """
    Model ki uporablja za atribute povrečno doc2vec predstavitev besed v dokumentu
    """
    if doc2vec is None:
        doc2vec = Doc2vecLearner()
        doc2vec.load_model(model_file='model/twitter1.d2v')
    train_X, train_Y, test_X, test_Y = read_train_test_data(doc2vec, mode="word2vec")
    print("Learn")
    if classifier is None:
        classifier = LogisticRegression(class_weight='balanced')
    classifier.fit(train_X, train_Y)
    print("Predict")
    prediction = [classifier.predict(i) for i in test_X]
    print("Dev-test")
    measurement(test_Y, prediction)
    if test:
        test_X, test_Y = read_train_test_data_test(doc2vec, mode="word2vec")
        prediction = [classifier.predict(i) for i in test_X]
        print("test")
        measurement(test_Y, prediction)


def end_test():
    """
    Končno testiranje modelov
    """
    doc2vec = Doc2vecLearner()
    doc2vec.load_model(model_file='model/twitter1.d2v')
    tfidf = TfIdf()
    tfidf.learn_tfidf(stop_words='english', max_df=0.8)

    print("Doc2vec")
    print("logistic")
    basic_model(doc2vec=doc2vec, classifier=LogisticRegression(class_weight='balanced'), test=True)

    print("ann")
    basic_model(doc2vec=doc2vec, classifier=MLPClassifier(random_state=1, hidden_layer_sizes=(33,)), test=True)

    print("Doc2vec_word")
    print("logistic")
    doc2vec_wordvec_model(doc2vec=doc2vec, classifier=LogisticRegression(class_weight='balanced'), test=True)

    print("ann")
    doc2vec_wordvec_model(doc2vec=doc2vec, classifier=MLPClassifier(random_state=1, hidden_layer_sizes=(33,)), test=True)

    print("tf_idf")
    print("logistic")
    tfidf_model(tfidf=tfidf, classifier=LogisticRegression(class_weight='balanced'), test=True)

    print("ann")
    tfidf_model(tfidf=tfidf, classifier=MLPClassifier(random_state=1, hidden_layer_sizes=(33,)), test=True)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # basic_model()
        # tfidf_model()
        # basic_model_ann()
        # basic_tfidf_model()
        # word2vec_model()
        # word2vec_model_tfidf()
        end_test()