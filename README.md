## In src.data_ingestion.py

at the end we can either create a run method or we can create an object like below
obj = DataIngestion()
obj.download_data_from_gcp()
obj.split_data()

insted of above we are creating run method

## TO track the models with MLFLOW , we will replace the below code in model_training.py

Code to be replaced :
def run(self):
""" Runs the entire model training process.
"""
try:
logger.info("Starting the model training process")
X_train, X_test, y_train, y_test = self.load_and_split_data()
lgbm_model = self.train_lgbm_model(X_train, y_train)
evaluation_metrics = self.evaluate_model(lgbm_model, X_test, y_test)
self.save_model(lgbm_model)

        logger.info(f"Model training completed with evaluation metrics: {evaluation_metrics}")
        #return evaluation_metrics
    except Exception as e:
        logger.error(f"Error in the model training process: {e}")
        raise CustomException("Error in running the model training process", e)

code replace with
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
