from sklearn.linear_model import LogisticRegression

from doc2vec import Doc2vecLearner
from utils import read_train_test_data, measurement
from tfidf import TfIdf
from sklearn.neural_network import MLPClassifier
import numpy as np
import warnings


def basic_model():
    # positive:  0.43190661478599224
    # neutral:  0.3191126279863481
    # negative:  0.43414634146341463
    # F1:  0.4436525755728229
    # Accuracy:  0.41304347826086957
    # macro-avarage:  0.395055194745
    doc2vec = Doc2vecLearner()
    doc2vec.load_model(model_file='model/twitter1.d2v')
    train_X, train_Y, test_X, test_Y = read_train_test_data(doc2vec)
    print("Learn")
    classifier = LogisticRegression(class_weight='balanced')
    classifier.fit(train_X, train_Y)
    print("Predict")
    prediction = []
    for i in test_X:
        prediction.append(classifier.predict(i))
    measurement(test_Y, prediction)


def tfidf_model():
    # positive recall:  0.4260700389105058
    # neutral recall:  0.3822525597269625
    # negative recall:  0.6292682926829268
    # F1:  0.5201284657686893
    # Accuracy:  0.492852888624
    # Macro-avg:  0.479196963773
    tfidf = TfIdf()
    tfidf.learn_tfidf(stop_words='english', max_df=0.8)
    print("Read data")
    train_X, train_Y, test_X, test_Y = read_train_test_data(tfidf, mode="tfidf")
    print("Learn")
    classifier = LogisticRegression(class_weight='balanced')
    classifier.fit(train_X, train_Y)
    print("Predict")
    prediction = []
    for i in test_X:
        prediction.append(classifier.predict(i))
    print(len(prediction))
    measurement(test_Y, prediction)


def basic_tfidf_model():
    # positive recall:  0.42412451361867703
    # neutral recall:  0.4453924914675768
    # negative recall:  0.5772357723577236
    # F1:  0.508788377864082
    # Accuracy:  0.483918999404
    # Macro - avg:  0.482250925815
    doc2vec = Doc2vecLearner()
    doc2vec.load_model(model_file='model/twitter1.d2v')
    print("tf-idf")
    tfidf = TfIdf()
    tfidf.learn_tfidf(stop_words='english', max_df=0.8)
    print("Read data")
    train_X, train_Y, test_X, test_Y = read_train_test_data(model1=doc2vec, model2=tfidf, mode="both")
    print("Learn")
    classifier = LogisticRegression(class_weight='balanced')
    classifier.fit(train_X, train_Y)
    print("Predict")
    prediction = []
    for i in test_X:
        prediction.append(classifier.predict(i))
    print(len(prediction))
    measurement(test_Y, prediction)


def word2vec_model():
    # positive recall:  0.4662775616083009
    # neutral recall:    0.5802047781569966
    # negative recall:    0.45934959349593496
    # F1:  0.5045331675574031
    # Accuracy:  0.483621203097
    # Macro - avg:  0.501943977754
    doc2vec = Doc2vecLearner()
    doc2vec.load_model(model_file='model/twitter1.d2v')
    train_X, train_Y, test_X, test_Y = read_train_test_data(doc2vec, mode="word2vec")
    print("Learn")
    classifier = LogisticRegression(class_weight='balanced')
    classifier.fit(train_X, train_Y)
    print("Predict")
    prediction = []
    for i in test_X:
        prediction.append(classifier.predict(i))
    print(len(prediction))
    measurement(test_Y, prediction)


def word2vec_model_ann():
    # positive recall:  0.4779507133592737
    # neutral recall:  0.3703071672354949
    # negative recall:  0.5772357723577236
    # F1:  0.5266286050015255
    # Accuracy:  0.49553305539
    # Macro - avg:  0.475164550984
    doc2vec = Doc2vecLearner()
    doc2vec.load_model(model_file='model/twitter1.d2v')
    train_X, train_Y, test_X, test_Y = read_train_test_data(doc2vec, mode="word2vec")
    print("Learn")
    classifier = MLPClassifier(random_state=1, hidden_layer_sizes=(33,))
    classifier.fit(train_X, train_Y)
    print("Predict")
    prediction = []
    for i in test_X:
        prediction.append(classifier.predict(i))
    measurement(test_Y, prediction)


def word2vec_model_tfidf():
    doc2vec = Doc2vecLearner()
    doc2vec.load_model(model_file='model/twitter1.d2v')
    print("tf-idf")
    tfidf = TfIdf()
    tfidf.learn_tfidf(stop_words='english', max_df=0.8)
    print("Read data")
    train_X, train_Y, test_X, test_Y = read_train_test_data(model1=doc2vec, model2=tfidf, mode="word2vec_tfidf")
    print("Learn")
    classifier = LogisticRegression(class_weight='balanced')
    classifier.fit(train_X, train_Y)
    print("Predict")
    prediction = []
    for i in test_X:
        prediction.append(classifier.predict(i))
    print(len(prediction))
    measurement(test_Y, prediction)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # basic_model()
        # tfidf_model()
        # basic_model_ann()
        # basic_tfidf_model()
        # word2vec_model()
        word2vec_model_tfidf()
