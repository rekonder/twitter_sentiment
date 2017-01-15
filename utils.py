import csv

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from preprocessor import ParseTweets
from scipy.sparse import hstack, coo_matrix

def count_classes(data="data/english/data_twits/Subtask_A_clean/dev_test.txt"):
    count = {"positive": 0, "negative": 0, "neutral": 0}
    with open(data, "r") as fil:
        try:
            reader = csv.reader(fil, delimiter='\t')
            for row in reader:
                count[row[1]] += 1
        except:
            pass
    print(count)


def sentiment_to_number(x):
    if x == "positive":
        return 1
    elif x == "negative":
        return -1
    else:
        return 0


def read_train_test_data(model1, model2=None, mode="doc2vec", train="data/english/data_twits/Subtask_A_clean/train.txt",
                         test="data/english/data_twits/Subtask_A_clean/dev_test.txt"):
    train_X, train_Y, test_X, test_Y = [], [], [], []
    train_X2, test_X2 = [], []
    pt = ParseTweets()
    print("Read training")
    if mode == "word2vec_tfidf":
        features = model2.tfidf.get_feature_names()
    with open(train, "r") as fil:
        try:
            reader = csv.reader(fil, delimiter='\t')
            for row in reader:
                if mode == "doc2vec":
                    train_X.append(model1.doctext_to_vec(row[2]))
                elif mode == "tfidf":
                    train_X.append(pt.tokenize_one_line(row[2]))
                elif mode == "word2vec":
                    train_X.append(model1.doctext_to_word2vec(row[2]))
                elif mode == "word2vec_tfidf":
                    train_X.append(model1.doctext_to_word2vec_with_tfidf(row[2], model2, features))
                elif mode == "both":
                    train_X.append(model1.doctext_to_vec(row[2]))
                    train_X2.append(pt.tokenize_one_line(row[2]))
                train_Y.append(sentiment_to_number(row[1]))
        except:
            pass
    print(len(train_X))
    if mode == "tfidf":
        train_X = model1.transform_tfidf(train_X)
    elif mode == "both":
        train_X2 = model2.transform_tfidf(train_X2)
        train_X = hstack([coo_matrix(train_X), coo_matrix(train_X2)]).tocsr()
    print("Read dev_test")
    with open(test, "r") as fil:
        reader = csv.reader(fil, delimiter='\t')
        for row in reader:
            if mode == "doc2vec":
                test_X.append(model1.doctext_to_vec(row[2]))
            elif mode == "tfidf":
                test_X.append(pt.tokenize_one_line(row[2]))
            elif mode == "word2vec":
                test_X.append(model1.doctext_to_word2vec(row[2]))
            elif mode == "word2vec_tfidf":
                test_X.append(model1.doctext_to_word2vec_with_tfidf(row[2], model2, features))
            elif mode == "both":
                test_X.append(model1.doctext_to_vec(row[2]))
                test_X2.append(pt.tokenize_one_line(row[2]))
            test_Y.append(sentiment_to_number(row[1]))
    print(len(test_Y))
    if mode == "tfidf":
        test_X = model1.transform_tfidf(test_X)
    elif mode == "both":
        test_X2 = model2.transform_tfidf(test_X2)
        test_X = hstack([coo_matrix(test_X), coo_matrix(test_X2)]).tocsr()
    return train_X, train_Y, test_X, test_Y

def measurement(real, prediction):
    recall_p, recall_neg, recall_neu, precision_p, precision_neg, precision_neu = 0, 0, 0, 0, 0, 0
    r_p, r_neg, r_neu, p_p, p_neg, p_neu = 0, 0, 0, 0, 0, 0
    for index, value in enumerate(prediction):
        if real[index] == 1:
            recall_p += 1
            if prediction[index] == 1:
                r_p += 1
        elif real[index] == 0:
            recall_neg += 1
            if prediction[index] == 0:
                r_neg += 1
        elif real[index] == -1:
            recall_neu += 1
            if prediction[index] == -1:
                r_neu += 1

        # precision
        if prediction[index] == 1:
            precision_p += 1
            if real[index] == 1:
                p_p += 1
        elif prediction[index] == 0:
            precision_neg += 1
            if real[index] == 0:
                p_neg += 1
        elif prediction[index] == -1:
            precision_neu += 1
            if real[index] == -1:
                p_neu += 1

    print("positive recall: ", r_p / recall_p)
    print("neutral recall: ", r_neu / recall_neu)
    print("negative recall: ", r_neg / recall_neg)

    rec_positive = r_p / recall_p
    prec_postive = p_p / precision_p
    rec_negative = r_neg / recall_neg
    prec_negative = p_neg / precision_neg
    print("F1: ", ((2 * rec_positive * prec_postive / (rec_positive + prec_postive)) +
                   (2 * rec_negative * prec_negative / (rec_negative + prec_negative))) / 2)
    print("Accuracy: ", accuracy_score(real, prediction))
    print("Macro-avg: ", recall_score(real, prediction, average='macro'))