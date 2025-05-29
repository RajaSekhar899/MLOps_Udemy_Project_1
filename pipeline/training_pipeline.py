from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.model_training import ModelTraining
from utils.common_functions import read_yaml
from config.path_config import *


if __name__ == "__main__":

    # 1. Data Ingestion
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    # 2. Data Preprocessing
    data_preprocessor = DataPreprocessing(TRAIN_FILE_PATH, TEST_FILE_PATH, PREPROCESSED_DIR, CONFIG_PATH)
    data_preprocessor.process()

    # 3. Model Training
    model_trainer = ModelTraining(PREPROCESSED_TRAIN_FILE_PATH, 
                                  PREPROCESSED_TEST_FILE_PATH, 
                                  MODEL_OUTPUT_PATH, 
                                  CONFIG_PATH)
    model_trainer.run()