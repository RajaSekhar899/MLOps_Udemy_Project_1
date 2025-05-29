import os

########################### DATA INGESTION #########################

RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR,"raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR,"train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR,"test.csv")

CONFIG_PATH = "config/config.yaml"


########################### DATA Preprocessing #########################
PREPROCESSED_DIR = "artifacts/preprocessed"
PREPROCESSED_TRAIN_FILE_PATH = os.path.join(PREPROCESSED_DIR, "train_preprocessed.csv")
PREPROCESSED_TEST_FILE_PATH = os.path.join(PREPROCESSED_DIR, "test_preprocessed.csv")

########################### MODEL TRAINING #########################
MODEL_OUTPUT_PATH = "artifacts/models/lgbm_model.pkl"

