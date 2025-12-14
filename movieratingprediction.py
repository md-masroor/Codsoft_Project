
# MOVIE RATING PREDICTION
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load Dataset
data = pd.read_csv("IMDb Movies India.csv", encoding='latin-1')
print("✅ Dataset Loaded Successfully\n")
print(data.head())

# 3. Select Required Columns
data = data[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Duration', 'Votes', 'Rating']]

# 4. Handle Missing Values
data['Genre'] = data['Genre'].fillna(data['Genre'].mode()[0])
data['Director'] = data['Director'].fillna(data['Director'].mode()[0])
data['Actor 1'] = data['Actor 1'].fillna(data['Actor 1'].mode()[0])
data['Actor 2'] = data['Actor 2'].fillna(data['Actor 2'].mode()[0])
data['Actor 3'] = data['Actor 3'].fillna(data['Actor 3'].mode()[0])

# Convert 'Duration' to numeric by removing ' min' and coercing to numeric
data['Duration'] = data['Duration'].astype(str).str.replace(' min', '', regex=False)
data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce')
data['Duration'] = data['Duration'].fillna(data['Duration'].median())

# Remove rows with missing Rating
data = data.dropna(subset=['Rating'])

# 5. Convert Votes to Numeric
data['Votes'] = data['Votes'].astype(str).str.replace(',', '')
data['Votes'] = pd.to_numeric(data['Votes'], errors='coerce')
data['Votes'] = data['Votes'].fillna(data['Votes'].median())

# 6. Encode Categorical Columns
le = LabelEncoder()
data['Genre'] = le.fit_transform(data['Genre'])
data['Director'] = le.fit_transform(data['Director'])
data['Actor 1'] = le.fit_transform(data['Actor 1'])
data['Actor 2'] = le.fit_transform(data['Actor 2'])
data['Actor 3'] = le.fit_transform(data['Actor 3'])

# 7. Feature Selection
X = data[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Duration', 'Votes']]
y = data['Rating']

# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Train Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 10. Predictions
y_pred = model.predict(X_test)

# 11. Model Evaluation
print("\n📊 Model Performance")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# 12. Visualization
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Movie Ratings")
plt.show()

# 13. Predict Rating for a New Movie
new_movie = pd.DataFrame(
    [[3, 1200, 500, 800, 1000, 150, 25000]],
    columns=['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Duration', 'Votes']
)

predicted_rating = model.predict(new_movie)
print("\n🎬 Predicted Movie Rating:", round(predicted_rating[0], 2))
