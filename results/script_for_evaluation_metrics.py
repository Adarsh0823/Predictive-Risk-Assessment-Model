from sklearn.metrics import classification_report, roc_auc_score

# Load the trained model
with open('models/risk_assessment_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Evaluate the model
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# Save evaluation metrics
with open('results/evaluation_metrics.txt', 'w') as file:
    file.write("Model Evaluation Metrics:\n\n")
    file.write(report)
    file.write(f"ROC-AUC: {roc_auc}\n")

print("Evaluation metrics saved successfully!")
