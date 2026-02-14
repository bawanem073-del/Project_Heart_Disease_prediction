# Heart Disease Prediction using Machine Learning
# Project Overview

This project predicts whether a person has heart disease or not using Machine Learning classification algorithms.

We implemented and compared three models:

Logistic Regression

Random Forest Classifier

Decision Tree Classifier

The goal is to evaluate and compare model performance using classification metrics.

# Dataset Information

The dataset contains medical attributes such as:

Chest pain type

Resting Blood Pressure (BP)

EKG results

Maximum Heart Rate (Max HR)

ST depression

Number of vessels fluro

Thallium

Heart Disease (Target Variable)

Target Variable:

0 → No Heart Disease

1 → Heart Disease

# Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

# Project Workflow

Data Loading & Exploration

Checked shape, null values, and statistics

Visualized distributions using histograms and boxplots

Data Preprocessing

Label Encoding for target variable

Feature scaling using StandardScaler

Used ColumnTransformer inside Pipeline

Model Training

Logistic Regression

Random Forest Classifier

Decision Tree Classifier

Model Evaluation

Accuracy Score

Precision Score

Recall Score

F1 Score

Confusion Matrix

# Model Evaluation Results
# Logistic Regression

Accuracy Score: 0.9074

Precision Score: 0.9444

Recall Score: 0.8095

F1 Score: 0.8718

 #Random Forest Classifier

Accuracy Score: 0.8148

Precision Score: 0.7895

Recall Score: 0.7143

F1 Score: 0.7500

# Decision Tree Classifier

Accuracy Score: 0.7222

Precision Score: 0.6250

Recall Score: 0.7143

F1 Score: 0.6667

# Best Performing Model

# Logistic Regression performed the best among all models with:

Highest Accuracy (90.74%)

Highest Precision (94.44%)

Strong F1 Score (87.18%)

This indicates that Logistic Regression provides the most balanced and reliable predictions for this dataset.

# Confusion Matrix

Confusion matrices were plotted for:

Logistic Regression

Random Forest Classifier

They help visualize:

True Positives

True Negatives

False Positives

False Negatives

# Conclusion

Logistic Regression outperformed both Random Forest and Decision Tree models in this project.
