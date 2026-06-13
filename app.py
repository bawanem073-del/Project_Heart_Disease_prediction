
from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import Customdata, PredictPipeline


application = Flask(__name__)

app = application


app.secret_key = "cardiovision_secret_key"

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
       
        prediction = predict_pipeline.predict(pred_df)[0]
        probability = predict_pipeline.predict_proba(pred_df)
        risk_score = round(probability[0][1] * 100, 2)
        confidence = round(max(probability[0]) * 100, 2)

        age = int(request.form.get('age'))
        sex = int(request.form.get('sex'))
        bp = float(request.form.get('bp'))
        cholesterol = float(request.form.get('cholesterol'))
        max_hr = float(request.form.get('max_hr'))
        exercise_angina = int(request.form.get('exercise_angina'))

        gender = "Male" if sex == 1 else "Female"


        risk_factors = []

        if bp > 140:
            risk_factors.append("High Blood Pressure")

        if cholesterol > 240:
            risk_factors.append("High Cholesterol")

        if max_hr < 120:
            risk_factors.append("Low Maximum Heart Rate")

        if exercise_angina == 1:
            risk_factors.append("Exercise Angina Present")

        if len(risk_factors) == 0:
            risk_factors.append("No Major Risk Factors Detected")


        if prediction == 1:

            recommendations = [
                "Consult a cardiologist",
                "Monitor blood pressure regularly",
                "Reduce cholesterol intake",
                "Follow a heart healthy diet"
            ]

        else:

            recommendations = [
                "Continue regular exercise",
                "Maintain balanced diet",
                "Routine health screening",
                "Monitor BP periodically"
            ]

        # Store report data in session

        session['prediction'] = int(prediction)

        session['risk_score'] = risk_score
        session['confidence'] = confidence

        session['risk_factors'] = risk_factors
        session['recommendations'] = recommendations

        session['age'] = age
        session['bp'] = bp
        session['cholesterol'] = cholesterol
        session['max_hr'] = max_hr
        session['gender'] = gender


        return redirect(url_for('report'))
    

@app.route('/report')
def report():

    if session.get('prediction') is None:
        return redirect(url_for('predict_datapoint'))

    return render_template(
        'report.html',
        prediction=session.get('prediction'),
        risk_score=session.get('risk_score'),
        confidence=session.get('confidence'),
        risk_factors=session.get('risk_factors'),
        recommendations=session.get('recommendations'),
        age=session.get('age'),
        bp=session.get('bp'),
        cholesterol=session.get('cholesterol'),
        max_hr=session.get('max_hr'),
        gender=session.get('gender')
    )



@app.route('/about')
def about():
    return render_template('about.html')



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)    
