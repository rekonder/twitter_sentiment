from preprocessor import ParseTweets
from os import listdir
from os.path import join
import csv


def save_all_tweets_in_one_file():
    for data in ["data/corpus/positive.txt", "data/corpus/negative.txt"]:
        print(data)
        with open(data, "r") as tf:
            lines = tf.readlines()
            tt = ParseTweets()
            for text in lines:
                new_text = tt.tokenize_one_line(text)
                if tt.getLanguage(new_text) == 'en':
                    with open("data/corpus/all_english1.txt", "a") as tf:
                        tf.write(new_text + "\n")


def clean_dataset():
    path = "data/english/data_twits/Subtask_A"
    path_clean = "data/english/data_twits/Subtask_A_clean"
    for i in listdir(path):
        if i.endswith("semeval_tweets.txt"):
            print(i)
            with open(join(path, i), "r") as fil:
                try:
                    reader = csv.reader(fil, delimiter='\t')
                    for row in reader:
                        if row[2] != "Not Available":
                            with open(join(path_clean, i), "a") as tf:
                                tf.write("\t".join(row) + "\n")
                except:
                    pass


def join_dataset():
    #path = "data/english/data_twits/Subtask_A"
    path = "data/english/data_twits/Subtask_A_clean"
    for i in listdir(path):
        if not i.startswith("twitter-2016") or i == "twitter-2016train-A_semeval_tweets.txt":
            print(i, "train")
            with open(join(path, i), "r") as fil:
                try:
                    reader = csv.reader(fil, delimiter='\t')
                    for row in reader:
                        with open("data/english/data_twits/Subtask_A_clean/train.txt", "a") as tf:
                            tf.write("\t".join(row) + "\n")
                except:
                    pass
        elif i == "twitter-2016dev-A_semeval_tweets.txt" or i == "twitter-2016devtest-A_semeval_tweets.txt":
            print(i, "dev_test")
            with open(join(path, i), "r") as fil:
                try:
                    reader = csv.reader(fil, delimiter='\t')
                    for row in reader:
                        with open("data/english/data_twits/Subtask_A_clean/dev_test.txt", "a") as tf:
                            tf.write("\t".join(row) + "\n")
                except:
                    pass
        else:
            print(i, "test")
            with open(join(path, i), "r") as fil:
                try:
                    reader = csv.reader(fil, delimiter='\t')
                    for row in reader:
                        with open("data/english/data_twits/Subtask_A_clean/test.txt", "a") as tf:
                            tf.write("\t".join(row) + "\n")
                except:
                    pass