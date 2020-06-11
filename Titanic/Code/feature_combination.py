#-*- coding utf8 -*-
"""
@author:
@brief: combine feature
"""
import logging
import optparse
import cofig
import os
import numpy as np
from utils import pickle
import pandas as pd

def main():
    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="mantian.log", filemode='a', level=logging.DEBUG,
                        datefmt="%m/%d/%y %H:%M:%S")
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt="%m/%d/%y %H:%M%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logger = logging.getLogger(__name__)

    dTrain = pickle._load(cofig.CLEANED_DATA_TRAIN)
    dTest = pickle._load(cofig.CLEANED_DATA_TEST)
    target = np.array(dTrain['Survived'])
    dTrain.drop('Survived', inplace=True)

    feature_train = np.array(dTrain)
    feature_test = np.array(dTest)

    del dTrain, dTest

    logger.info("feature generate start")
    for dir in os.listdir(cofig.DATA_FEATURE_DIR):
        feature_dir = os.path.join(cofig.DATA_FEATURE_DIR, dir)
        for feature_file_or_dir in os.listdir(feature_dir):
            if os.path.isdir(os.path.join(feature_dir, feature_file_or_dir)):
                logger.info("feature %s generate start"%feature_file_or_dir)
                for feature_files in os.listdir(feature_file_or_dir):
                    for file in feature_files:
                        data_path = os.path.join(os.path.join(feature_file_or_dir, feature_files), file)
                        if file == 'train.pkl':
                            train = pickle._load(data_path)
                            feature_train = np.hstack([feature_train, train])
                        else:
                            test = pickle._save(data_path)
                            feature_test = np.hstack([feature_test, test])
                logger.info("feature %s generate end"%feature_file_or_dir)
            else:
                logger.info("feature %s generate start"%feature_dir)
                for feature_file in os.listdir(feature_file_or_dir):
                    for file in feature_files:
                        data_path = os.path.join(os.path.join(feature_file_or_dir, feature_files), file)
                        if file == 'train.pkl':
                            train = pickle._load(data_path)
                            feature_train = np.hstack([feature_train, train])
                        else:
                            test = pickle._save(data_path)
                            feature_test = np.hstack([feature_test, test])
                logger.info("feature %s generate end"%feature_dir)

    pickle._save(cofig.FINISH_DATA_TRAIN, feature_train)
    pickle._save(cofig.FINISH_DATA_TESTM, feature_test)
    pickle._save(cofig.FINISH_DATA_TARGET, target)

if __name__ == '__main__':
    main()



