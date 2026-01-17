

from sklearn.linear_model import LogisticRegression
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

pipeline.fit(X_train, y_train)

predict_hd = pipeline.predict(X_test)


# #data scaling
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

# #model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# predict_hd = model.predict(X_test)

acs = accuracy_score(y_test, predict_hd)
pcs = precision_score(y_test, predict_hd)
res = recall_score(y_test, predict_hd)
fs = f1_score(y_test, predict_hd)
cm = confusion_matrix(y_test, predict_hd)

print("Accuracy Score : ",acs)
print("Precision Score: ", pcs)
print("Recall Score : ", res)
print("F1 Score : ", fs)

# Plot heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix Heatmap")
plt.show()


