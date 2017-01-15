import string
import nltk
from langdetect import detect
from guess_language import guess_language
import enchant
import re

tweets_tokens = {}
tweets_lans = {}


class ParseTweets:
    def __init__(self):
        pass

    def tokenize_one_line(self, text, join=False):
        text = self.parseUsers(text)
        text = self.parseLinks(text)
        text = self.parseHashtags(text)
        text = self.parseEmoticons(text)
        # remove punctuation
        table = text.maketrans({key: None for key in string.punctuation})
        text = text.translate(table)
        tokens = nltk.word_tokenize(text)
        return " ".join(tokens) if not join else (tokens, " ".join(tokens))

    def tokenize(self, file):
        with open(file, "r") as tf:
            for text in tf.readlines():
                print("orig:", text, "tokens:", self.tokenize_one_line(text))
                print("")

    def parseUsers(self, text):
        """
        izbrise uporabnike (@user)
        """
        return re.sub('\@[\w]*', '', text)

    def parseLinks(self, text):
        """
        Brisanje linkov v tweetu
        """
        return re.sub('http[\w\:\/\.]*', '', text)

    def parseHashtags(self, text):
        """
        Brisanje hastagov v tweetih
        """
        return " ".join([tag.strip('#') if tag.startswith("#") else tag for tag in text.split()])

    def parseEmoticons(self, text):
        """
        Brisanje emoticone v tweetih
        """
        POSITIVE = ["*O", "*-*", "*O*", "*o*", "* *",
                    ":P", ":D", ":d", ":p", ";P", ";D", ";d", ";p",
                    ":-)", ";-)", ":=)", ";=)", ":<)", ":>)", ";>)", ";=)",
                    "=}", ":)", "(:;)", "(;", ":}", "{:", ";}", "{;:]",
                    "[;", ":')", ";')", ":-3", "{;", ":]",
                    ";-3", ":-x", ";-x", ":-X", ";-X", ":-}", ";-=}", ":-]", ";-]", ":-.)",
                    "^_^", "^-^",
                    ":-)", ":)", ":o)", ":]", ":3", ":c)", ":>", "=]", "8)", "=)", ":}", ":^)", ";)", ";-)",
                    ":')", ":P", ":-P", ":p", ";P", ";p", ":d", ";D", ";d", ":D", ":-D", "8-D", "8D", "x-D",
                    "xD", "X-D", "XD", "=-D", "2=2D", "=-3", "=3", "B^D", "&lt;3",
                    ]

        NEGATIVE = [":(", ";(", ":'(", ":'(", ":(", ":-(",
                    "=(", "={", "):", ");",
                    ")':", ")';", ")=", "}=",
                    ";-{{", ";-{", ":-{{", ":-{",
                    ":-(", ";-(", ":,)", ":'{", "[:", ";]",
                    ]

        pattern2 = "|".join(map(re.escape, POSITIVE + NEGATIVE))
        text = re.sub(pattern2, '', text)

        # odstrani tudi znakovne emoticone (slikce)
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U00002702-\U000027B0"  # other symbols
                                   u"\U000024C2-\U0001F251"  # enclosed characters
                                   u"\U0001F30D-\U0001F567"  # additional symbols
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        return text

    def getLanguage(self, text):
        """
        Ugotovi jezik tweeta
        """
        lan = guess_language(text)
        return 'en' if lan != 'ar' and lan != 'fa' else lan

if __name__ == "__main__":
    tt = ParseTweets()
