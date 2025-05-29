
import os
import sys
import pandas as pd
import numpy as np
from utils.common_functions import read_yaml, load_data
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE



logger = get_logger(__name__)

class DataPreprocessing:
    def __init__(self, train_file_path=TRAIN_FILE_PATH, test_file_path=TEST_FILE_PATH,
                 preprocessed_dir=PREPROCESSED_DIR, config_path=CONFIG_PATH):
        """ Initializes the DataPreprocessing class.
        Args:
            train_file_path (str): Path to the training data file.
            test_file_path (str): Path to the testing data file.
            preprocessed_dir (str): Directory where preprocessed data will be saved.
            config_path (str): Path to the configuration file.  
        """
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.preprocessed_dir = preprocessed_dir
        self.config = read_yaml(config_path) 

        os.makedirs(self.preprocessed_dir, exist_ok=True)

        logger.info("Data Preprocessing initialized")

    def preprocess_data(self,df):
        try:
            logger.info("Starting data preprocessing")

            # Dropping unnecessary columns
            df.drop(columns = ['Unnamed: 0', 'Booking_ID'], inplace=True)
            df.drop_duplicates(inplace=True) 

            logger.info("Dropped unnecessary columns")

            cat_columns = self.config['data_preprocessing']['categorical_columns']
            num_columns = self.config['data_preprocessing']['numerical_columns']

            # Label encoding for categorical variables
            logger.info("Encoding the categorical variables")

            label_encoder = LabelEncoder()
            mappings = {}

            for col in cat_columns:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
                

            logger.info("Label encoding mappings:")
            
            for col, mapping in mappings.items():
                logger.info(f"{col}: {mapping}")
            
            logger.info("Label encoding is done for the categorical variables")

            logger.info("Skewness treatment for numerical variables")

            skewness_threshold = self.config['data_preprocessing']['skewness_threshold']
            for col in num_columns:
                skewness = df[col].skew()
                if abs(skewness) > skewness_threshold:
                    df[col] = np.log1p(df[col])  # Apply log transformation
                    logger.info(f"Applied log transformation to {col} due to skewness: {skewness}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise CustomException("Failed to preprocess data", e)
        
    def handle_imbalanced_data(self, df):
        try:
            logger.info("Handling imbalanced data using SMOTE")
            X = df.drop(columns=self.config['data_preprocessing']['target_column'])
            y = df[self.config['data_preprocessing']['target_column']]

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            resampled_df[self.config['data_preprocessing']['target_column']] = y_resampled

            logger.info("Imbalanced data handled successfully")
            return resampled_df
        
        except Exception as e:
            logger.error(f"Error while handling imbalanced data: {e}")
            raise CustomException("Failed to handle imbalanced data", e)
        
    def feature_selection(self, df):
        try:
            logger.info("Starting feature selection")
            target_column = self.config['data_preprocessing']['target_column']

            X = df.drop(columns=target_column)
            y = df[target_column]

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            feature_importance = model.feature_importances_
            logger.info("Feature importances calculated")

            feature_importance_df = pd.DataFrame({
                                                    'Feature': X.columns, 
                                                    'Importance': feature_importance
                                                    }).sort_values(by='Importance', ascending=False)
            
            no_of_features = self.config['data_preprocessing']['no_of_features']
            top_10_features = feature_importance_df['Feature'].head(no_of_features).values

            top_10_df = df[top_10_features.tolist() + [target_column]]

            logger.info(f"Top {no_of_features} features selected: {top_10_features.tolist()}")
            logger.info("Feature selection completed successfully")
            return top_10_df
        
        except Exception as e:
            logger.error(f"Error during feature selection: {e}")
            raise CustomException("Failed to select features", e)
        
    def save_preprocessed_data(self, df, file_path_name):
        try:
            logger.info("Saving preprocessed data")
            df.to_csv(file_path_name, index=False)
            logger.info(f"Preprocessed data saved to {file_path_name}")

        except Exception as e:
            logger.error(f"Error while saving preprocessed data: {e}")
            raise CustomException("Failed to save preprocessed data", e)
        
    def process(self):
        try:
            logger.info("Loading data from RAW directory")

            # Load training and testing data
            train_df = load_data(self.train_file_path)
            test_df = load_data(self.test_file_path)

            # Preprocess training and testing data
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            # Handle imbalanced data
            train_df = self.handle_imbalanced_data(train_df)


            # Feature selection
            train_df = self.feature_selection(train_df)
            test_df = test_df[train_df.columns]
            

            # Save preprocessed data
            self.save_preprocessed_data(train_df, PREPROCESSED_TRAIN_FILE_PATH)
            self.save_preprocessed_data(test_df, PREPROCESSED_TEST_FILE_PATH)

            logger.info("Data processing pipeline completed successfully")

        except Exception as e:
            logger.error(f"Error in data preprocessing pipeline: {e}")
            raise CustomException("Data preprocessing failed", e)


if __name__ == "__main__":
    try:
        data_preprocessor = DataPreprocessing(TRAIN_FILE_PATH, TEST_FILE_PATH, PREPROCESSED_DIR, CONFIG_PATH)
        data_preprocessor.process()
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}")
        
        



        

            