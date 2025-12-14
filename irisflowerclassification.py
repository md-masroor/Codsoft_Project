
# IRIS FLOWER CLASSIFICATION
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Dataset
data = pd.read_csv("IRIS (1).csv")
print("✅ Dataset Loaded Successfully\n")
print(data.head())

# 3. Rename Columns (if needed)
data.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

# 4. Encode Target Variable
le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])

# 5. Feature Selection
X = data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = data['Species']

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)

# 9. Model Evaluation
print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))

print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
plt.imshow(cm)
plt.colorbar()
plt.xticks(range(3), le.classes_)
plt.yticks(range(3), le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 10. Predict Species for a New Flower
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
)

prediction = model.predict(new_flower)
predicted_species = le.inverse_transform(prediction)

print("\n🌸 Predicted Iris Species:", predicted_species[0])
