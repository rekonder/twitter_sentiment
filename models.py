from sklearn.linear_model import LogisticRegression

from doc2vec import Doc2vecLearner
from utils import read_train_test_data, measurement


def basic_model():
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

if __name__ == "__main__":
    basic_model()