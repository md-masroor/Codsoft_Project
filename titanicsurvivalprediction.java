# TITANIC SURVIVAL PREDICTION 
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Dataset (Upload Titanic-Dataset.csv manually)
data = pd.read_csv("Titanic-Dataset.csv")
print("✅ Dataset Loaded Successfully\n")
print(data.head())

# 3. Handle Missing Values (CLEAN METHOD)
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# 4. Drop Unnecessary Columns
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# 5. Encode Categorical Columns
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])       # male=1, female=0
data['Embarked'] = le.fit_transform(data['Embarked'])

# 6. Select Features and Target
X = data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = data['Survived']

# 7. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 9. Make Predictions
y_pred = model.predict(X_test)

# 10. Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Model Accuracy:", accuracy)

print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 11. Predict Survival for a New Passenger (NO WARNINGS)
new_passenger = pd.DataFrame(
    [[3, 1, 25, 15, 2]],
    columns=['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
)

prediction = model.predict(new_passenger)

print("\n📌 New Passenger Prediction:")
if prediction[0] == 1:
    print("🟢 Passenger Survived")
else:
    print("🔴 Passenger Did Not Survive")
