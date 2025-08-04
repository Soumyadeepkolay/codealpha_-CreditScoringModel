import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


# Generate a binary classification dataset with financial-like features
X, y = make_classification(n_samples=1000,
                           n_features=6,
                           n_informative=4,
                           n_redundant=0,
                           random_state=42)

# Convert to DataFrame
df = pd.DataFrame(X, columns=['income', 'debt', 'credit_score', 'age', 'num_loans', 'loan_amount'])
df['default'] = y

# Feature Engineering: add debt-to-income ratio
df['debt_income_ratio'] = df['debt'] / (df['income'] + 1e-5)  # avoid divide-by-zero


# Define features and target
features = ['income', 'debt', 'credit_score', 'age', 'num_loans', 'loan_amount', 'debt_income_ratio']
X = df[features]
y = df['default']

# Split data into training and testing sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for many models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)



# Predict class labels
y_pred = model.predict(X_test_scaled)

# Predict class probabilities
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Print classification metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))


# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.2f})", color="blue")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Credit Scoring Model")
plt.legend()
plt.grid(True)
plt.show()


# Plot feature importance
importances = model.feature_importances_
feature_names = features

plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Random Forest - Feature Importance")
plt.grid(True)
plt.show()

