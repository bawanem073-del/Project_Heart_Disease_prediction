
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds


        except Exception  as e:
            raise CustomException
        

class Customdata:
    def __init__(self,
             age: int,
             sex: int,
             chest_pain_type: int,
             bp: float,
             cholesterol: float,
             fbs_over_120: int,
             ekg_results: int,
             max_hr: float,
             exercise_angina: int,
             st_depression: float,
             slope_of_st: int,
             number_of_vessels_fluro: int,
             thallium: int):
        
        self.age = age
        self.sex = sex
        self.chest_pain_type = chest_pain_type
        self.bp = bp
        self.cholesterol = cholesterol
        self.fbs_over_120 = fbs_over_120
        self.ekg_results = ekg_results
        self.max_hr = max_hr
        self.exercise_angina = exercise_angina
        self.st_depression = st_depression
        self.slope_of_st = slope_of_st
        self.number_of_vessels_fluro = number_of_vessels_fluro
        self.thallium = thallium        


    def get_data_as_data_frame(self):

        try:
            custom_data_input_dict = {
                "Age": [self.age],
                "Sex": [self.sex],
                "Chest pain type": [self.chest_pain_type],
                "BP": [self.bp],
                "Cholesterol": [self.cholesterol],
                "FBS over 120": [self.fbs_over_120],
                "EKG results": [self.ekg_results],
                "Max HR": [self.max_hr],
                "Exercise angina": [self.exercise_angina],
                "ST depression": [self.st_depression],
                "Slope of ST": [self.slope_of_st],
                "Number of vessels fluro": [self.number_of_vessels_fluro],
                "Thallium": [self.thallium]
            }

            return pd.DataFrame(custom_data_input_dict )
        

        except Exception as e:
            raise CustomException(e,sys)  
        