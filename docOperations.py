import csv
from os import listdir
from os.path import join

from preprocessor import ParseTweets


def save_all_tweets_in_one_file():
    """
    Shrani samo angleške tweete
    """
    for data in ["data/corpus/positive.txt", "data/corpus/negative.txt"]:
        print(data)
        with open(data, "r") as tf:
            tt = ParseTweets()
            for text in tf.readlines():
                new_text = tt.tokenize_one_line(text)
                if tt.getLanguage(new_text) == 'en':
                    with open("data/corpus/all_english1.txt", "a") as tf:
                        tf.write(new_text + "\n")


def clean_dataset():
    """
    Predprocesiraj vsaki tweet
    """
    path = "data/english/data_twits/Subtask_A"
    for i in listdir(path):
        if i.endswith("semeval_tweets.txt"):
            print(i)
            with open(join(path, i), "r") as fil:
                try:
                    for row in csv.reader(fil, delimiter='\t'):
                        if row[2] != "Not Available":
                            with open(join("data/english/data_twits/Subtask_A_clean", i), "a") as tf:
                                tf.write("\t".join(row) + "\n")
                except:
                    pass


def join_dataset():
    """
    Zdreževanje podatkovnih zbirk na učno, testno in testno-učno množico
    """
    path = "data/english/data_twits/Subtask_A_clean"
    for i in listdir(path):
        if not i.startswith("twitter-2016") or i == "twitter-2016train-A_semeval_tweets.txt":
            print(i, "train")
            with open(join(path, i), "r") as fil:
                try:
                    for row in csv.reader(fil, delimiter='\t'):
                        with open("data/english/data_twits/Subtask_A_clean/train.txt", "a") as tf:
                            tf.write("\t".join(row) + "\n")
                except:
                    pass
        elif i == "twitter-2016dev-A_semeval_tweets.txt" or i == "twitter-2016devtest-A_semeval_tweets.txt":
            print(i, "dev_test")
            with open(join(path, i), "r") as fil:
                try:
                    for row in csv.reader(fil, delimiter='\t'):
                        with open("data/english/data_twits/Subtask_A_clean/dev_test.txt", "a") as tf:
                            tf.write("\t".join(row) + "\n")
                except:
                    pass
        else:
            print(i, "test")
            with open(join(path, i), "r") as fil:
                try:
                    for row in csv.reader(fil, delimiter='\t'):
                        with open("data/english/data_twits/Subtask_A_clean/test.txt", "a") as tf:
                            tf.write("\t".join(row) + "\n")
                except:
                    pass
