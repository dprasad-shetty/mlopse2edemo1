import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
# Create sample dataset

# Load the dataset
df = pd.read_csv('heart.csv')

# Define features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


C = 0.1
solver = 'lbfgs'
penalty = 'l2'
# Train a Logistic Regression model # RF # XGboost
model = LogisticRegression(C=C, solver=solver, penalty=penalty,max_iter=1000)  # Increased max_iter for convergence
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


model_filename = "logistic_regression_model.pkl"
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")