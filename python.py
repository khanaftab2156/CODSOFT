import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay

df = pd.read_csv("C:/Users/khana/OneDrive/Desktop/TITANIC SURVIVAL PREDICTION/Titanic-Dataset.csv")
print(df.head())

#checking missing values
print("before filling")
print(df.isnull().sum())

#filling missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Cabin'] = df['Cabin'].fillna('unknown')
df.replace([np.inf, -np.inf], np.nan, inplace=True)

#remove duplicates
df.drop_duplicates(inplace=True)

#checking missing after filling
print("after filling")
print(df.isnull().sum())

#data encoding
LE = LabelEncoder()
df['Sex'] = LE.fit_transform(df['Sex'])
df['Embarked'] = LE.fit_transform(df['Embarked'])

#feature(X) and target(Y) select
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
Y = df['Survived']

#split train-Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#training model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

#for prediction
Y_pred = model.predict(X_test)

#check accuracy for the model
acc = accuracy_score(Y_test, Y_pred)
print("Accuracy:",acc)

#for graphical representation
df['Survived'].value_counts().plot(kind = 'bar')
plt.title("Survival No")
plt.show()

#confusion metrix

C = confusion_matrix(Y_test, Y_pred)
print(C)

ConfusionMatrixDisplay(confusion_matrix=C).plot()
plt.show()

#user input
pclass = int(input("Passenger class "))
age = int(input("Age "))
pred = model.predict([[pclass,1,age,0,0,7.25,2]])
print(pred)
