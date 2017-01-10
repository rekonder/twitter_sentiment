import csv

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score


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


def read_train_test_data(model, train="data/english/data_twits/Subtask_A_clean/train.txt",
                         test="data/english/data_twits/Subtask_A_clean/dev_test.txt"):
    train_X, train_Y, test_X, test_Y = [], [], [], []
    print("Read training")
    with open(train, "r") as fil:
        try:
            reader = csv.reader(fil, delimiter='\t')
            for row in reader:
                train_X.append(model.doctext_to_vec(row[2]))
                train_Y.append(sentiment_to_number(row[1]))
        except:
            pass
    print("Read dev_test")
    with open(test, "r") as fil:
        try:
            reader = csv.reader(fil, delimiter='\t')
            for row in reader:
                test_X.append(model.doctext_to_vec(row[2]))
                test_Y.append(sentiment_to_number(row[1]))
        except:
            pass
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