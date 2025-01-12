# Predictive Risk Assessment Model

## Project Overview
This project develops a machine learning model to predict healthcare members at high risk of hospitalization. The goal is to enable proactive interventions, improving healthcare outcomes and reducing costs.

---

## Objectives
- Predict high-risk members based on demographic and health data.
- Provide actionable insights to healthcare management teams.
- Enable targeted care plans to reduce hospitalizations.

---

## Features
- Data cleaning and preprocessing using Python and PySpark.
- Logistic regression model with performance metrics.
- Deployment-ready Python scripts for risk prediction.

---

## Tools and Technologies
- **Programming Languages:** Python, PySpark
- **Libraries:** pandas, scikit-learn, matplotlib, seaborn
- **Cloud Services:** AWS RDS
- **Modeling:** Logistic regression

---

## Data Description
| Column         | Description                      |
|----------------|----------------------------------|
| MemberID       | Unique identifier for members    |
| Age            | Member's age in years           |
| Gender         | Member's gender                 |
| BloodPressure  | Blood pressure levels (mmHg)    |
| Cholesterol    | Cholesterol levels (mg/dL)      |
| Diabetes       | Binary indicator for diabetes   |
| Smoker         | Binary indicator for smoking    |
| HeartDisease   | Binary indicator for heart disease |
| HighRisk       | Target variable (1 = High Risk) |

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Predictive-Risk-Assessment-Model.git
   cd Predictive-Risk-Assessment-Model

## Instal Dependencies
2. pip install -r requirements.txt

## Usage
Train the Model: Run train_model.py to train and save the logistic regression model.
Make Predictions: Use predict_risk.py to predict hospitalization risk for new members.

## Results
Accuracy: 85%
ROC-AUC: 0.91
Enabled proactive care planning for high-risk members.

## Future Enhancements
Incorporate advanced models like XGBoost or Random Forest.
Add real-time prediction capability via APIs.
Expand dataset to include more health metrics.

