
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import Customdata, PredictPipeline


application = Flask(__name__)

app = application

MODEL_ACCURACY = 88.89

@app.route('/')
def index():

    return render_template('index.html', accuracy=MODEL_ACCURACY)


@app.route('/predictdata', methods=['GET', 'POST'])



def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = Customdata(
            age=int(request.form.get('age')),
            sex=int(request.form.get('sex')),
            chest_pain_type=int(request.form.get('chest_pain_type')),
            bp=float(request.form.get('bp')),
            cholesterol=float(request.form.get('cholesterol')),
            fbs_over_120=int(request.form.get('fbs_over_120')),
            ekg_results=int(request.form.get('ekg_results')),
            max_hr=float(request.form.get('max_hr')),
            exercise_angina=int(request.form.get('exercise_angina')),
            st_depression=float(request.form.get('st_depression')),
            slope_of_st=int(request.form.get('slope_of_st')),
            number_of_vessels_fluro=int(request.form.get('number_of_vessels_fluro')),
            thallium=int(request.form.get('thallium'))
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)    
