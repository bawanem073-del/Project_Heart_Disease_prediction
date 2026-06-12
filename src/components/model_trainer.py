
import os
import sys

from dataclasses import dataclass
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerCongig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerCongig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.info("Data split completed")

            models =  {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "KNN": KNeighborsClassifier(),
                "CatBoost": CatBoostClassifier(verbose=0),
                "XGBoost": XGBClassifier(),

                } 
            


            params = {

                    "Logistic Regression": {
                        'C': [0.001, 0.01, 0.1, 1, 10, 100]
                    },

                    "Decision Tree": {
                        'criterion': ['gini', 'entropy', 'log_loss'],
                        'max_depth': [None, 5, 10, 20, 30],
                        'min_samples_split': [2, 5, 10, 20],
                        'min_samples_leaf': [1, 2, 4, 8]
                    },

                    "Random Forest": {
                        'n_estimators': [50, 100, 200, 300],
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },

                    "Gradient Boosting": {
                        'learning_rate': [0.001, 0.01, 0.05, 0.1],
                        'n_estimators': [50, 100, 200, 300],
                        'subsample': [0.6, 0.8, 1.0],
                        'max_depth': [3, 5, 7]
                    },

                    "AdaBoost": {
                        'n_estimators': [50, 100, 200, 300],
                        'learning_rate': [0.001, 0.01, 0.1, 1.0]
                    },

                    "KNN": {
                        'n_neighbors': [3, 5, 7, 9, 11],
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan', 'minkowski']
                    },
                    "XGBoost": {
                        'learning_rate': [0.01, 0.1],
                        'n_estimators': [100, 200],
                        'max_depth': [3, 5, 7]
                    },

                    "CatBoost": {
                        'depth': [4, 6, 8],
                        'learning_rate': [0.01, 0.1],
                        'iterations': [100, 200]
                    }
                }


            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            # to get the best model score from dict

            best_model_score = max(model_report.values())

            # To get the best model name from dict

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]


            if best_model_score < 0.6:
                raise CustomException("No best model found")
            

            logging.info(f"Best found model on both training and testing dataset is {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            acc_score = accuracy_score(y_test, predicted)
            

            return acc_score 

        except Exception as e:
            raise CustomException(e, sys)