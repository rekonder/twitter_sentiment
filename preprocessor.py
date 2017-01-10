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

    def tokenize_one_line(self, text):
        text = self.parseUsers(text)
        text = self.parseLinks(text)
        text = self.parseHashtags(text)
        text = self.parseEmoticons(text)
        # remove punctuation
        table = text.maketrans({key: None for key in string.punctuation})
        text = text.translate(table)
        tokens = nltk.word_tokenize(text)
        return " ".join(tokens)

    def tokenize(self, file):
        with open(file, "r") as tf:
            lines = tf.readlines()
            for text in lines:
                new_text = self.tokenize_one_line(text)
                print("orig:", text, "tokens:", new_text)
                print("")

    def parseUsers(self, text):
        # izbrise uporabnike (@user)
        # ce je treba bolj obdelat: re.findall() ... in potem obdelas najdene
        text = re.sub('\@[\w]*', '', text)
        return text

    def parseLinks(self, text):
        # izbrise povezave (http...)
        # ce je treba bolj obdelat: re.findall() ... in potem obdelas najdene
        text = re.sub('http[\w\:\/\.]*', '', text)

        return text

    def parseHashtags(self, text):
        # izbrise hashtage (#hashtag)
        # ce je treba bolj obdelat: re.findall() ... in potem obdelas najdene
        # text = re.sub('\#[\w]*', '', text)  # vse hashtage zbrise

        # naprednejsa: izbrise # pri hashtagih
        #hashtags = re.findall('\#[\w]*', text)  # najde vse hashtage
        #text = re.sub('\#[\w]*', '[\w]', text)  # vse hashtage zbrise
        text = " ".join([tag.strip('#') if tag.startswith("#") else tag for tag in text.split()])

        #for h in hashtags:
        #    h = h[1:]  # odstrani # (nicti znak v nizu)
        #    text += (h + " ")  # doda hashtag brez # nazaj v niz

        return text

    def parseEmoticons(self, text):
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

        # "&lt;3" je <3 je srce

        NEGATIVE = [":(", ";(", ":'(", ":'(", ":(", ":-(",
                    "=(", "={", "):", ");",
                    ")':", ")';", ")=", "}=",
                    ";-{{", ";-{", ":-{{", ":-{",
                    ":-(", ";-(", ":,)", ":'{", "[:", ";]",
                    ]

        # lahko tudi pozitivne in negativne posebej in jih zamenjas s cim drugim
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
        # izbrise uporabnike (@user)
        # text = re.sub('@[\w]*', '', text)
        # izbrise povezave (http...)
        # text = re.sub('http[\w\:\/\.]*', '', text)
        # text = self.parseEmoticons(text)
        lan = guess_language(text)
        if lan != 'ar' and lan != 'fa':
            lan = 'en'
        return lan

if __name__ == "__main__":
    tt = ParseTweets()

    # zaenkrat samo izpisuje tokene
    #tt.tokenize("data/corpus/positive.txt")

    # zaenkrat samo izpisuje jezik NOT WORK
    #tt.getLanguage("data/corpus/positive.txt")
