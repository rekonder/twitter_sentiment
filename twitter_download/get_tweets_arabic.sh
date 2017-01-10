#!/bin/bash

python download_tweets_user_api.py --dist ../data/arabic/DOWNLOAD/DEV/SemEval2017-task4-dev.subtask-A.arabic.txt --output ../data/arabic/data_twits/SemEval2017-task4-dev.subtask-A.arabic --user

python download_tweets_user_api.py --dist ../data/arabic/DOWNLOAD/TRAIN-ONLY/SemEval2017-task4-train-only.subtask-A.arabic.txt --output ../data/arabic/data_twits/SemEval2017-task4-train-only.subtask-A.arabic --user
