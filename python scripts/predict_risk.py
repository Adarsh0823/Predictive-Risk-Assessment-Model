import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import pickle

# Load data
data = pd.read_csv('data/synthetic_health_data.csv')
X = data.drop(columns=['HighRisk'])
y = data['HighRisk']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Save model
with open('models/risk_assessment_model.pkl', 'wb') as file:
    pickle.dump(model, file)
