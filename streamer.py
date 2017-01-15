import json
import string

import nltk
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

access_token = "Dodaj"
access_token_secret = "Dodaj"
consumer_key = "Dodaj"
consumer_secret = "Dodaj"


class Listener(StreamListener):
    _my_count = 0
    # Ce zelis pozitivne twitte nasatavi positive na True in odkomentiraj pozitivne keywords spodaj
    # sicer nastavi na False in  odkomentiraj negativne keywords spodaj
    _positive = True

    def on_data(self, data):
        if "limit" not in data:
            if not json.loads(data)["text"].replace("\n", " ").startswith("RT"):
                self._my_count += 1
                if self._positive:
                    print(self._my_count, " positive")
                    with open("data/corpus/positive.txt", "a") as tf:
                            tf.write(json.loads(data)["text"].replace("\n", " ") + "\n")
                else:
                    print(self._my_count, " negative")
                    with open("data/corpus/negative.txt", "a") as tf:
                        tf.write(json.loads(data)["text"].replace("\n", " ") + "\n")
        return True

    def on_error(self, status):
        print("error: ",  status)


class TwitterKeywords:
    """
    Class s katerim sem želel pridobit določene besede v arabščin, prebrati emoticone iz datoteke
    """
    def __init__(self):
        self.keywords = []

    def read_data(self):
        with open("data/corpus/tweets_arabic.txt", "r") as tf:
            lines = tf.readlines()
            for text in lines:
                text = "".join(filter(lambda x: x == " " or x not in string.printable, text)).strip()
                table = text.maketrans({key: None for key in string.punctuation})
                text = text.translate(table)
                tokens = nltk.word_tokenize(text)
                for x in tokens:
                    if x not in self.keywords:
                        self.keywords.append(x)
        print(self.keywords)

    def emoji(self):
        with open("data/corpus/emoji.txt", "r") as tf:
            lines = tf.readlines()
            for num, text in enumerate(lines):
                if num % 2 == 1:
                    self.keywords.append(text.strip())
        return self.keywords[300:]

if __name__ == "__main__":
    # Za zbiranje pozitivnih
    keywords = [":)", ":D", "=)", ";)", ":-)", ";-)", ":-D", "=D", ";P", "=]"]

    # za zbiranje negativnih
    # keywords = [":(", ":')", ":'(", ":-(", "D:", ";(", ":-/", ":|"]

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    twitterStream = Stream(auth, Listener())
    twitterStream.filter(track=keywords, languages=["ar", "en"])