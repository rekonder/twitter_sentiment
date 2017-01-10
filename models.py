from sklearn.linear_model import LogisticRegression

from doc2vec import Doc2vecLearner
from utils import read_train_test_data, measurement
from tfidf import TfIdf


def basic_model():
    # positive:  0.43190661478599224
    # neutral:  0.3191126279863481
    # negative:  0.43414634146341463
    # Accurancy:  0.41304347826086957
    # F1:  0.4436525755728229
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

if __name__ == "__main__":
    # basic_model()
    tfidf_model()
