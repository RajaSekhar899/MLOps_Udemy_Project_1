import pandas as pd
import numpy as np
from utils.common_functions import read_yaml, load_data
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
import os
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from src.data_preprocessing import DataPreprocessing
from config.model_params import *
from scipy.stats import randint, uniform
import joblib
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)
class ModelTraining:
    def __init__(self, preprocessed_train_file_path=PREPROCESSED_TRAIN_FILE_PATH,
                 preprocessed_test_file_path=PREPROCESSED_TEST_FILE_PATH,
                 model_output_path=MODEL_OUTPUT_PATH, config_path=CONFIG_PATH):
        """ Initializes the ModelTraining class.
        Args:
            preprocessed_train_file_path (str): Path to the preprocessed training data file.
            preprocessed_test_file_path (str): Path to the preprocessed testing data file.
            model_output_path (str): Directory where the trained model will be saved.
            config_path (str): Path to the configuration file.
        """
        self.preprocessed_train_file_path = preprocessed_train_file_path
        self.preprocessed_test_file_path = preprocessed_test_file_path
        self.model_output_path = model_output_path
        self.config = read_yaml(config_path)

        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        """ Loads and splits the preprocessed data into training and testing sets.
        Returns:
            X_train (pd.DataFrame): Features for training.
            X_test (pd.DataFrame): Features for testing.
            y_train (pd.Series): Target variable for training.
            y_test (pd.Series): Target variable for testing.
        """
        try:
            #logger.info(f"Loading training data from {self.preprocessed_train_file_path}")
            logger.info(f"Loading training data from {self.preprocessed_train_file_path.replace(os.sep, '/')}")
            train_df = load_data(self.preprocessed_train_file_path)

            #logger.info(f"Loading testing data from {self.preprocessed_test_file_path}")
            logger.info(f"Loading testing data from {self.preprocessed_test_file_path.replace(os.sep, '/')}")
            test_df = load_data(self.preprocessed_test_file_path)

            target_column = self.config['data_preprocessing']['target_column']

            X_train = train_df.drop(columns=target_column)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=target_column)
            y_test = test_df[target_column]

            logger.info("Data loaded and split into training and testing sets")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error in loading and splitting data: {e}")
            raise CustomException(f"Error in loading and splitting data: {e}")
    
    def train_lgbm_model(self, X_train, y_train):
        try:
            logger.info("Starting model training")
            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params['random_state'])

            logger.info("Performing Randomized Search for hyperparameter tuning") 
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                n_jobs=self.random_search_params['n_jobs'],
                scoring=self.random_search_params['scoring']
            )

            logger.info("Starting the hyper parameter tuning process")
            # Fit the model using RandomizedSearchCV
            random_search.fit(X_train, y_train)
            logger.info("Hyper Parameter tuning process is completed")
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            logger.info(f"Best hyperparameters are: {best_params}")

            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise CustomException(f"Error in model training: {e}")
        
    def evaluate_model(self, model, X_test, y_test):
        """ Evaluates the trained model on the test set.
        Args:
            model: The trained LightGBM model.
            X_test (pd.DataFrame): Features for testing.
            y_test (pd.Series): Target variable for testing.
        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        try:
            logger.info("Evaluating the model")
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            logger.info(f"Model evaluation completed with accuracy: {accuracy},F1 score: {f1}, Precision: {precision}, Recall: {recall}")

            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall
            }
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            raise CustomException(f"Error in model evaluation: {e}")
        
    def save_model(self, model):
        """ Saves the trained model to the specified output path.
        Args:
            model: The trained LightGBM model.
        """
        try:
            
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info(f"Saving the model to {self.model_output_path}")
            joblib.dump(model, self.model_output_path)
            logger.info("Model saved successfully")

        except Exception as e:
            logger.error(f"Error in saving the model: {e}")
            raise CustomException(f"Error in saving the model: {e}")
        
    def run(self):
        """ Runs the entire model training process.
        """
        try:
            with mlflow.start_run():
                
                logger.info("Starting the model training process")
                logger.info("Starting the MLFLOW experiment")
                logger.info("logging the training and testing dataset to MLFLOW")

                mlflow.log_artifact(self.preprocessed_train_file_path, artifact_path="datasets")
                mlflow.log_artifact(self.preprocessed_test_file_path, artifact_path="datasets")

                X_train, X_test, y_train, y_test = self.load_and_split_data()
                lgbm_model = self.train_lgbm_model(X_train, y_train)
                evaluation_metrics = self.evaluate_model(lgbm_model, X_test, y_test)
                self.save_model(lgbm_model)

                logger.info("Logging the model to MLFLOW")
                mlflow.log_artifact(self.model_output_path)

                logger.info("Logging the params and evaluation metrics to MLFLOW")
                mlflow.log_params(lgbm_model.get_params())
                mlflow.log_metrics(evaluation_metrics)

                logger.info(f"Model training completed with evaluation metrics: {evaluation_metrics}")
            
        except Exception as e:
            logger.error(f"Error in the model training process: {e}")
            raise CustomException("Error in running the model training process", e)
        
if __name__ == "__main__":
    model_trainer = ModelTraining(PREPROCESSED_TRAIN_FILE_PATH, 
                                  PREPROCESSED_TEST_FILE_PATH, 
                                  MODEL_OUTPUT_PATH, 
                                  CONFIG_PATH)
    model_trainer.run()
    