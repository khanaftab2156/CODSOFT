import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn.metrics import classification_report

#dataset load
dataset = pd.read_csv("C:/Users/khana/OneDrive/Desktop/Iris flower classification/IRIS.csv")
print(dataset.head())

#data preprocessing
print(dataset.isnull().sum())

#remove duplicate
dataset.drop_duplicates(inplace = True)

#data encoding
LE = LabelEncoder()
dataset["species"] = LE.fit_transform(dataset["species"])
print(dataset.head())
print(dataset.tail())

#feature(x) and target(y) select
x = dataset[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = dataset["species"]

#split train-test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state  = 42)

#training the model
model = RandomForestClassifier( n_estimators=100, random_state = 42)
model.fit(x_train, y_train)

#for prediction
y_pred = model.predict(x_test)

#accuracy check
acc = accuracy_score(y_test, y_pred)
print("Accuracy:",acc)

#classification report
print("classification report is :")
print(classification_report(y_test, y_pred))

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

ConfusionMatrixDisplay(cm).plot()
plt.show()

#prediction
pred = model.predict([[6.3, 3.3, 6.0, 2.5]])
print(pred)

importance = model.feature_importances_
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
plt.bar(features, importance)
plt.title("feature importance")
plt.xlabel("features")
plt.ylabel("importance")
plt.show()

