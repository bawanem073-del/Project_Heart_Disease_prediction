
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("C:/Users/LOQ/OneDrive/Desktop/health_data.csv")

print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.describe())

sns.countplot(x = "Heart Disease", data = df)
plt.show()

plt.hist(df["Age"], edgecolor = "black")
plt.title("Age Distribution")
plt.show()

plt.hist(df["Cholesterol"], color = "powderblue", edgecolor = "dimgray")
plt.title("Cholesterol Distribution", fontsize = 14)
plt.xlabel("Cholesterol ", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()

plt.hist(df["BP"], color = "lightcoral", edgecolor = "darkred", bins = 20)
plt.title("Resting Blood Pressure Distribution")
plt.xlabel("Resting Blood Pressure")
plt.ylabel("Frequency")
plt.show()

sns.boxplot(x = "Heart Disease", y = "Age", data = df)
plt.title("Age vs Heart Disease")
plt.show()


# Label encoding
le = LabelEncoder()
df["Heart Disease"] = le.fit_transform(df["Heart Disease"])

X = df[["Chest pain type", "BP", "EKG results", "Max HR", "ST depression", "Number of vessels fluro", "Thallium"]]
y = df["Heart Disease"]

#data spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

num_feature = ["Chest pain type", "BP", "EKG results", "Max HR", "ST depression", "Number of vessels fluro", "Thallium"]

#Pipeline Implementation

trf1 = ColumnTransformer([("num", StandardScaler(), num_feature)])

pipeline = Pipeline([("preprocessing", trf1),
                     ("Classifier", LogisticRegression())])

pipeline2 = Pipeline([("preprocessing", trf1),
                      ("Classifier", RandomForestClassifier(n_estimators=100, random_state=42))])

pipeline3 = Pipeline([("preprocessing", trf1),
                      ("Clssifier", DecisionTreeClassifier( random_state=42))])                      

pipeline.fit(X_train, y_train)
pipeline2.fit(X_train, y_train)
pipeline3.fit(X_train, y_train)

predict_hd = pipeline.predict(X_test)
predict_rfc = pipeline2.predict(X_test)
predict_dtc = pipeline3.predict(X_test)
 

print("Evaluation Metrics - Logistic Regression")
print("Accuracy Score : ",accuracy_score(y_test, predict_hd))
print("Precision Score: ", precision_score(y_test, predict_hd))
print("Recall Score : ", recall_score(y_test, predict_hd))
print("F1 Score : ", f1_score(y_test, predict_hd))

print("Evaluation Metrics - Random Forest Classifier")
print("Accuracy Score : ",accuracy_score(y_test, predict_rfc))
print("Precision Score: ", precision_score(y_test, predict_rfc))
print("Recall Score : ", recall_score(y_test, predict_rfc))
print("F1 Score : ", f1_score(y_test, predict_rfc))

print("Evaluation Metrics - Decision Tree  Classifier")
print("Accuracy Score : ",accuracy_score(y_test, predict_dtc))
print("Precision Score: ", precision_score(y_test, predict_dtc))
print("Recall Score : ", recall_score(y_test, predict_dtc))
print("F1 Score : ", f1_score(y_test, predict_dtc))


# Plot heatmap


plt.figure(figsize = (14, 5))
plt.subplot(1,2,1)

sns.heatmap(confusion_matrix(y_test, predict_hd), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix For Logistic Regression")

plt.subplot(1,2,2)

sns.heatmap(confusion_matrix(y_test, predict_rfc), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix For Random Forest Classifier")

plt.show()


